"""FastAPI application deployed to each agent VM."""

import asyncio
import base64
import hashlib
import hmac as _hmac
import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, field_validator

from baal_agent.compaction import maybe_compact
from baal_agent.config import AgentSettings
from baal_agent.context import build_dynamic_context, build_static_system_prompt, build_subagent_prompt, build_system_prompt
from baal_agent.database import AgentDatabase
from baal_agent.inference import InferenceClient
from baal_agent.security import MAX_SEND_FILE_SIZE, PathSecurityError, validate_workspace_path
from baal_agent.telegram_bot import TelegramBot
from baal_agent.plugins import PluginManager
from baal_agent.scheduler import CronScheduler
from baal_agent.tools import (
    configure_tools,
    execute_tool,
    get_tool_definitions,
    shutdown_browser,
    shutdown_code_executor,
    shutdown_mcp,
    shutdown_processes,
    shutdown_shell,
    start_code_executor,
    start_mcp,
    start_shell,
)

logger = logging.getLogger(__name__)

settings = AgentSettings()
db = AgentDatabase(db_path=settings.db_path)
inference = InferenceClient(api_key=settings.libertai_api_key)

_scheduler: CronScheduler | None = None
_plugin_manager: PluginManager | None = None
_telegram_bot: TelegramBot | None = None
_telegram_bot_task: asyncio.Task | None = None


async def _send_file_via_telegram(
    rel_path: str, caption: str = "", *, chat_id: str = ""
) -> None:
    """Best-effort file delivery via the agent's Telegram bot, if configured.

    Args:
        chat_id: Telegram chat ID to send to. Falls back to owner_telegram_id.
    """
    if _telegram_bot is None or not _telegram_bot.connected:
        return
    target = chat_id or settings.owner_telegram_id
    if not target:
        return
    try:
        await _telegram_bot._send_local_file(target, rel_path, caption)
    except Exception as e:
        logger.error(f"Telegram file delivery failed for {rel_path}: {e}")


# ── Subagent registry ─────────────────────────────────────────────────

MAX_CONCURRENT_SUBAGENTS = 5
DEFAULT_SUBAGENT_TIMEOUT = 300
MAX_SUBAGENT_TIMEOUT = 600
_SUBAGENT_RETENTION = 3600  # 1 hour


@dataclass
class SubagentRun:
    id: str
    label: str
    task: str
    persona: str | None
    status: str  # running / completed / failed / timeout
    chat_id: str
    started_at: float
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None
    asyncio_task: asyncio.Task | None = field(default=None, repr=False)


_subagent_runs: dict[str, SubagentRun] = {}


def _prune_old_subagent_runs():
    """Remove completed subagent runs older than _SUBAGENT_RETENTION."""
    cutoff = time.time() - _SUBAGENT_RETENTION
    to_remove = [
        run_id for run_id, run in _subagent_runs.items()
        if run.status != "running" and run.completed_at and run.completed_at < cutoff
    ]
    for run_id in to_remove:
        del _subagent_runs[run_id]


# ── Chat run registry ────────────────────────────────────────────────

_RUN_RETENTION = 300  # keep completed runs for 5 minutes


@dataclass
class ChatRun:
    chat_id: str
    task: asyncio.Task
    events: list[dict] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    done: bool = False
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    user_message: str | list[dict] = ""


_active_runs: dict[str, ChatRun] = {}  # keyed by chat_id


def _prune_old_chat_runs():
    """Remove completed chat runs older than _RUN_RETENTION."""
    cutoff = time.time() - _RUN_RETENTION
    to_remove = [
        chat_id for chat_id, run in _active_runs.items()
        if run.done and run.completed_at is not None and run.completed_at < cutoff
    ]
    for chat_id in to_remove:
        del _active_runs[chat_id]


# ── Telegram callback ────────────────────────────────────────────────

async def _telegram_agent_turn(
    message: str | list[dict], chat_id: str
) -> tuple[str | None, list[dict]]:
    """Callback for TelegramBot: run an agent turn and return (text, file_events).

    Registers a ChatRun in _active_runs so cancel_chat_run() works for /stop.
    """
    from baal_agent.telegram_bot import TELEGRAM_CHANNEL_HINT

    file_events: list[dict] = []

    # Register in _active_runs so /stop can cancel this turn
    run = ChatRun(
        chat_id=chat_id,
        task=None,  # type: ignore — set below
        user_message=message,
    )

    async def _do_turn():
        return await _run_agent_turn(
            message, chat_id, channel_hint=TELEGRAM_CHANNEL_HINT,
            file_events=file_events,
        )

    run.task = asyncio.create_task(_do_turn())
    _active_runs[chat_id] = run

    try:
        result = await run.task
    finally:
        run.done = True
        run.completed_at = time.time()
    return result, file_events


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _plugin_manager, _telegram_bot, _telegram_bot_task
    await db.initialize()
    configure_tools(settings.workspace_path, db=db, inference=inference, model=settings.model)
    await start_shell()
    await start_code_executor()
    await start_mcp(settings.mcp_servers)

    # Ensure workspace directories exist
    workspace = Path(settings.workspace_path)
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "skills").mkdir(parents=True, exist_ok=True)

    # Load plugins from workspace/plugins/
    _plugin_manager = PluginManager(settings.workspace_path)
    _plugin_manager.load_plugins()

    # Start cron scheduler (replaces old heartbeat loop).
    # It reads workspace/cron.json every tick and falls back to the legacy
    # HEARTBEAT.md behaviour when no cron.json exists.
    _scheduler = CronScheduler(settings.workspace_path, heartbeat_interval=settings.heartbeat_interval)
    await _scheduler.start(_run_cron_job)

    # Start Telegram bot if configured
    if settings.telegram_bot_token:
        _telegram_bot = TelegramBot(
            token=settings.telegram_bot_token,
            owner_telegram_id=settings.owner_telegram_id,
            db=db,
            agent_turn_callback=_telegram_agent_turn,
            cancel_run_callback=cancel_chat_run,
            workspace_path=settings.workspace_path,
        )
        try:
            await _telegram_bot.start()
            _telegram_bot_task = asyncio.create_task(_telegram_bot.poll_loop())
            logger.info("Telegram bot started")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            _telegram_bot = None

    yield

    if _telegram_bot_task and not _telegram_bot_task.done():
        _telegram_bot_task.cancel()
        try:
            await _telegram_bot_task
        except asyncio.CancelledError:
            pass
    if _telegram_bot:
        await _telegram_bot.stop()

    if _scheduler:
        await _scheduler.stop()
    # Cancel all active chat runs
    for run in _active_runs.values():
        if not run.done and not run.task.done():
            run.task.cancel()
    await shutdown_mcp()
    await shutdown_processes()
    await shutdown_browser()
    await shutdown_code_executor()
    await shutdown_shell()
    await db.close()


app = FastAPI(title=f"Baal Agent: {settings.agent_name}", lifespan=lifespan)


# ── Auth middleware ────────────────────────────────────────────────────

PUBLIC_PREFIXES = ("/dl/", "/assets/")
PUBLIC_PATHS = {"/health", "/", "/index.html", "/favicon.ico", "/manifest.json"}


@app.middleware("http")
async def verify_auth(request: Request, call_next):
    """Reject requests without a valid Bearer token, except public paths."""
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    token_hash = hashlib.sha256(token.encode()).hexdigest() if token else ""
    if not token or not secrets.compare_digest(token_hash, settings.agent_secret_hash):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


# ── CORS (registered last so it wraps auth as the outermost layer) ─────

if settings.local_ui_enabled:
    _cors_origins = [o.strip() for o in settings.local_ui_cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Content-Type"],
    )


# ── Core agentic loop ─────────────────────────────────────────────────

async def _run_agent_turn(
    message: str | list[dict],
    chat_id: str,
    *,
    restricted: bool = False,
    max_iterations: int | None = None,
    store_history: bool = True,
    file_events: list[dict] | None = None,
    system_prompt_override: str | None = None,
    channel_hint: str | None = None,
) -> str | None:
    """Run a single agentic turn (message -> tool loop -> response).

    Args:
        message: The user/system message to process.
        chat_id: Conversation identifier for history.
        restricted: If True, use restricted tool set (no spawn).
        max_iterations: Override max tool iterations.
        store_history: Whether to persist messages to DB.
        file_events: Optional accumulator for send_file events (heartbeat/subagent).
        system_prompt_override: If set, use this instead of building the default prompt.
        channel_hint: Optional formatting hint appended to the system prompt
            (e.g. Telegram formatting constraints).

    Returns:
        The final text response, or None if no text was generated.
    """
    iterations = max_iterations or settings.max_tool_iterations
    tools = get_tool_definitions(include_spawn=not restricted)
    tool_names = [t["function"]["name"] for t in tools]
    pending_images: list[dict] = []

    def _stash_images(blocks: list[dict]):
        pending_images.extend(blocks)

    if store_history:
        # Use static prompt + dynamic context injection for KV cache preservation.
        # Dynamic content (memory/skills) is injected near the end of the message
        # list so the prefix tokens stay identical across turns.
        static_prompt = build_static_system_prompt(
            settings.system_prompt,
            settings.agent_name,
            settings.workspace_path,
            tool_names=tool_names,
            heartbeat_interval=settings.heartbeat_interval,
            fqdn=settings.agent_fqdn,
        )
        dynamic_context = build_dynamic_context(settings.workspace_path)
        if channel_hint:
            dynamic_context = (dynamic_context + "\n\n" + channel_hint) if dynamic_context else channel_hint
        await db.add_message(chat_id, "user", message)
        messages = await maybe_compact(
            db, inference, chat_id, static_prompt, settings.model, settings,
            dynamic_context=dynamic_context,
        )
    else:
        # Non-cached path (heartbeat, subagents): use combined prompt in a single message
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
            if "Available tools:" not in system_prompt:
                system_prompt += f"\nAvailable tools: {', '.join(tool_names)}"
        else:
            system_prompt = build_system_prompt(
                settings.system_prompt,
                settings.agent_name,
                settings.workspace_path,
                tool_names=tool_names,
                heartbeat_interval=settings.heartbeat_interval,
                fqdn=settings.agent_fqdn,
            )
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": message})

    final_text = None
    total_tool_calls = 0

    for _iteration in range(iterations):
        assistant_msg = await inference.chat(
            messages=messages, model=settings.model, tools=tools
        )

        text_content = assistant_msg.content
        tool_calls = assistant_msg.tool_calls

        tc_for_db = None
        if tool_calls:
            total_tool_calls += len(tool_calls)
            tc_for_db = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]

        if store_history:
            await db.add_message(chat_id, "assistant", text_content, tool_calls=tc_for_db)

        assistant_dict: dict = {"role": "assistant"}
        if text_content:
            assistant_dict["content"] = text_content
        if tc_for_db:
            assistant_dict["tool_calls"] = tc_for_db
        messages.append(assistant_dict)

        if text_content:
            final_text = text_content

        if not tool_calls:
            # Auto-skill nudge (Task 1.4): if the turn used many tools,
            # ask the agent to consider saving the procedure as a skill.
            skill_text = await _maybe_skill_nudge(
                total_tool_calls, messages, chat_id, store_history, tools,
            )
            if skill_text:
                final_text = skill_text
            return final_text

        # Execute tool calls concurrently
        async def _exec_tool(tc):
            tool_name = tc.function.name
            arguments = tc.function.arguments

            # Plugin pre_tool hook
            if _plugin_manager is not None:
                arguments = await _plugin_manager.fire_modify(
                    "pre_tool", arguments, tool_name,
                )

            if tool_name == "spawn" and not restricted:
                result = await _handle_spawn(arguments, chat_id)
            else:
                result = await execute_tool(
                    tool_name, arguments,
                    image_callback=_stash_images,
                )

            # Plugin post_tool hook
            if _plugin_manager is not None:
                result = await _plugin_manager.fire_modify(
                    "post_tool", result, tool_name,
                )

            return result

        results = await asyncio.gather(
            *[_exec_tool(tc) for tc in tool_calls], return_exceptions=True
        )

        for tc, result in zip(tool_calls, results):
            if isinstance(result, BaseException):
                result = f"Error executing {tc.function.name}: {result}"

            # Detect send_file markers and accumulate for callers
            if isinstance(result, str) and result.startswith("__SEND_FILE__:"):
                parts = result.split(":", 2)
                rel_path = parts[1] if len(parts) > 1 else ""
                caption = parts[2] if len(parts) > 2 else ""
                if file_events is not None:
                    file_events.append({"path": rel_path, "caption": caption})
                result = f"File sent to user: {rel_path}"

            if store_history:
                await db.add_message(chat_id, "tool", result, tool_call_id=tc.id)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Inject any images collected from tool results as a user message
        if pending_images:
            messages.append({"role": "user", "content": list(pending_images)})
            pending_images.clear()

    return final_text


# ── Auto-skill nudge ─────────────────────────────────────────────────

async def _maybe_skill_nudge(
    total_tool_calls: int,
    messages: list[dict],
    chat_id: str,
    store_history: bool,
    tools: list[dict],
) -> str | None:
    """If the turn used many tools, nudge the agent to consider saving a skill.

    Runs one additional inference iteration so the agent can decide whether
    to create a skill file. Returns the agent's text response if any.
    """
    threshold = settings.auto_skill_threshold
    if threshold <= 0 or total_tool_calls < threshold:
        return None

    nudge = (
        f"[System] You completed a complex task with {total_tool_calls} tool calls. "
        f"If this procedure might be useful again, consider saving it as a skill at "
        f"workspace/skills/<name>/SKILL.md with YAML frontmatter (name, description)."
    )

    if store_history:
        await db.add_message(chat_id, "user", nudge)
    messages.append({"role": "user", "content": nudge})

    try:
        response = await inference.chat(
            messages=messages, model=settings.model, tools=tools
        )

        text = response.content
        tool_calls = response.tool_calls

        tc_for_db = None
        if tool_calls:
            tc_for_db = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]

        if store_history:
            await db.add_message(chat_id, "assistant", text, tool_calls=tc_for_db)

        # Append assistant message (with tool_calls) to messages list so
        # subsequent tool result messages have the correct conversation structure
        if tool_calls:
            assistant_dict: dict = {"role": "assistant"}
            if text:
                assistant_dict["content"] = text
            assistant_dict["tool_calls"] = tc_for_db
            messages.append(assistant_dict)

        # Execute any tool calls (the agent may write a skill file)
        if tool_calls:
            for tc in tool_calls:
                result = await execute_tool(tc.function.name, tc.function.arguments)
                if store_history:
                    await db.add_message(chat_id, "tool", result, tool_call_id=tc.id)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # Let the model see the tool results and produce a final response
            try:
                final = await inference.chat(messages=messages, model=settings.model, tools=None)
                if final.content and store_history:
                    await db.add_message(chat_id, "assistant", final.content)
                return final.content
            except Exception:
                pass  # If follow-up fails, return the original text

        return text
    except Exception as e:
        logger.warning(f"Auto-skill nudge inference failed: {e}")
        return None


# ── Spawn / subagent ──────────────────────────────────────────────────

async def _handle_spawn(arguments: str | dict, origin_chat_id: str) -> str:
    """Handle the spawn tool call — start a background subagent."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    task = arguments["task"]
    label = arguments.get("label", task[:50])
    persona = arguments.get("persona")
    timeout = min(int(arguments.get("timeout", DEFAULT_SUBAGENT_TIMEOUT)), MAX_SUBAGENT_TIMEOUT)

    _prune_old_subagent_runs()

    # Check concurrency limit for this chat
    active = sum(
        1 for r in _subagent_runs.values()
        if r.chat_id == origin_chat_id and r.status == "running"
    )
    if active >= MAX_CONCURRENT_SUBAGENTS:
        return f"Error: too many concurrent subagents ({active}/{MAX_CONCURRENT_SUBAGENTS}). Wait for some to finish."

    run_id = uuid.uuid4().hex[:8]
    run = SubagentRun(
        id=run_id,
        label=label,
        task=task,
        persona=persona,
        status="running",
        chat_id=origin_chat_id,
        started_at=time.time(),
    )
    _subagent_runs[run_id] = run

    # Emit spawned event
    await db.add_pending(
        origin_chat_id,
        json.dumps({"type": "subagent_spawned", "run_id": run_id, "label": label, "status": "running"}),
        source="subagent_event",
    )

    task_handle = asyncio.create_task(_run_subagent(run, timeout, origin_chat_id))
    run.asyncio_task = task_handle

    return (
        f"Subagent '{label}' spawned (id: {run_id}, timeout: {timeout}s). "
        f"The subagent is working in the background — do NOT repeat its task. "
        f"Move on to other work or inform the user you've delegated this task."
    )


async def _run_subagent(run: SubagentRun, timeout: int, origin_chat_id: str):
    """Run a subagent in the background with restricted tools and a lightweight prompt."""
    try:
        # Build lightweight subagent prompt
        tools = get_tool_definitions(include_spawn=False)
        tool_names = [t["function"]["name"] for t in tools]
        subagent_prompt = build_subagent_prompt(
            settings.agent_name,
            settings.workspace_path,
            tool_names=tool_names,
            persona=run.persona,
        )

        files: list[dict] = []
        result = await asyncio.wait_for(
            _run_agent_turn(
                run.task,
                chat_id=f"__subagent_{run.id}__",
                restricted=True,
                max_iterations=15,
                store_history=False,
                file_events=files,
                system_prompt_override=subagent_prompt,
            ),
            timeout=timeout,
        )

        run.status = "completed"
        run.result = result
        run.completed_at = time.time()

        await db.add_pending(
            origin_chat_id,
            f"[Task: {run.label}] {result or '(no output)'}",
            source="subagent",
        )
        # Emit completed event
        await db.add_pending(
            origin_chat_id,
            json.dumps({"type": "subagent_completed", "run_id": run.id, "label": run.label, "status": "completed"}),
            source="subagent_event",
        )
        # Forward file events (pending for web UI + direct Telegram delivery
        # only when the originating chat is itself Telegram — otherwise we'd
        # leak web-UI subagent files to the agent owner's Telegram).
        tg_target = origin_chat_id.removeprefix("tg:") if origin_chat_id.startswith("tg:") else ""
        for fe in files:
            await db.add_pending(
                origin_chat_id,
                json.dumps({"type": "file", "path": fe["path"], "caption": fe["caption"]}),
                source="subagent_file",
            )
            if tg_target:
                await _send_file_via_telegram(
                    fe["path"], fe.get("caption", ""), chat_id=tg_target,
                )

    except asyncio.TimeoutError:
        run.status = "timeout"
        run.error = f"Timed out after {timeout}s"
        run.completed_at = time.time()
        logger.warning(f"Subagent {run.id} ({run.label}) timed out after {timeout}s")
        await db.add_pending(
            origin_chat_id,
            json.dumps({"type": "subagent_failed", "run_id": run.id, "label": run.label, "status": "timeout", "error": run.error}),
            source="subagent_event",
        )

    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.completed_at = time.time()
        logger.error(f"Subagent {run.id} ({run.label}) failed: {e}")
        await db.add_pending(
            origin_chat_id,
            json.dumps({"type": "subagent_failed", "run_id": run.id, "label": run.label, "status": "failed", "error": str(e)}),
            source="subagent_event",
        )


# ── Cron job callback ────────────────────────────────────────────────

async def _run_cron_job(task: str, job_id: str):
    """Callback invoked by the CronScheduler for each due job."""
    chat_id = f"__cron_{job_id}__"
    files: list[dict] = []
    try:
        result = await _run_agent_turn(
            task,
            chat_id=chat_id,
            store_history=True,
            file_events=files,
        )
        for fe in files:
            await _send_file_via_telegram(fe["path"], fe.get("caption", ""))
        if result:
            logger.info(f"Cron job {job_id!r} produced output ({len(result)} chars)")
        else:
            logger.debug(f"Cron job {job_id!r}: no output")
    except Exception as e:
        logger.error(f"Cron job {job_id!r} failed: {e}")


# ── SSE helpers ───────────────────────────────────────────────────────

# Retryable inference errors (transient / server-side)
_RETRYABLE_ERRORS = (asyncio.TimeoutError, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)

# Loop-level retries on top of inference.py's own HTTP-level retries
_LOOP_INFERENCE_RETRIES = 2
_LOOP_RETRY_DELAY = 5  # seconds between loop-level retries


def _is_retryable(e: Exception) -> bool:
    """Check if an inference error is worth retrying at the loop level."""
    return isinstance(e, _RETRYABLE_ERRORS)

# Interval for SSE keepalive events during long-running operations.
# Sent as real data events (not SSE comments) so that reverse proxies
# (like the CRN *.2n6.me gateway) actually forward them.
_KEEPALIVE_INTERVAL = 15  # seconds


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_keepalive() -> str:
    """Real SSE data event to keep the connection alive through reverse proxies."""
    return f"data: {json.dumps({'type': 'keepalive'})}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str | list[dict]
    chat_id: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("message must not be empty")
            return v
        # list[dict] — validate structure
        if not v:
            raise ValueError("message list must not be empty")
        has_text = False
        image_count = 0
        for item in v:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError("each content block must have a 'type' field")
            if item["type"] == "text":
                has_text = True
            elif item["type"] == "image_url":
                image_count += 1
        if not has_text:
            raise ValueError("message must contain at least one text block")
        if image_count > 10:
            raise ValueError("message must not contain more than 10 images")
        return v


async def _emit(run: ChatRun, data: dict):
    """Append an SSE event to the run buffer and notify waiting readers.

    Injects `timestamp_ms` if the caller didn't set one — frontends use it
    to render per-message timestamps so all messages don't share the page's
    render time.
    """
    if "timestamp_ms" not in data:
        data["timestamp_ms"] = int(time.time() * 1000)
    async with run.condition:
        run.events.append(data)
        run.condition.notify_all()


async def _run_chat_background(run: ChatRun, chat_id: str, message: str | list[dict]):
    """Background agentic loop — emits SSE events to run.events buffer."""
    try:
        # Reload plugins so newly created plugins take effect immediately
        if _plugin_manager is not None:
            _plugin_manager.load_plugins()
            await _plugin_manager.fire("on_session_start", chat_id)

        tools = get_tool_definitions(include_spawn=True)
        tool_names = [t["function"]["name"] for t in tools]
        pending_images: list[dict] = []

        def _stash_images(blocks: list[dict]):
            pending_images.extend(blocks)

        # Use static prompt + dynamic context injection for KV cache preservation
        system_prompt = build_static_system_prompt(
            settings.system_prompt,
            settings.agent_name,
            settings.workspace_path,
            tool_names=tool_names,
            heartbeat_interval=settings.heartbeat_interval,
            fqdn=settings.agent_fqdn,
        )
        dynamic_context = build_dynamic_context(settings.workspace_path)

        await db.add_message(chat_id, "user", message)
        messages = await maybe_compact(
            db, inference, chat_id, system_prompt, settings.model, settings,
            dynamic_context=dynamic_context,
        )

        inference_timeout = settings.inference_timeout
        total_tool_calls = 0

        for _iteration in range(settings.max_tool_iterations):
            # Call inference with loop-level retry.
            assistant_msg = None
            last_inference_error = None

            for _inf_attempt in range(_LOOP_INFERENCE_RETRIES + 1):
                try:
                    assistant_msg = await asyncio.wait_for(
                        inference.chat(
                            messages=messages, model=settings.model, tools=tools
                        ),
                        timeout=inference_timeout,
                    )
                    break  # success — exit retry loop
                except Exception as e:
                    last_inference_error = e
                    if _is_retryable(e) and _inf_attempt < _LOOP_INFERENCE_RETRIES:
                        logger.warning(
                            f"Inference attempt {_inf_attempt + 1} failed "
                            f"({type(e).__name__}), retrying in {_LOOP_RETRY_DELAY}s"
                        )
                        await _emit(run, {
                            "type": "text",
                            "content": "\n\n*[Inference error, retrying...]*\n\n",
                        })
                        await asyncio.sleep(_LOOP_RETRY_DELAY)
                        continue
                    else:
                        break  # non-retryable or exhausted retries

            if assistant_msg is None:
                err_name = type(last_inference_error).__name__ if last_inference_error else "Unknown"
                if isinstance(last_inference_error, asyncio.TimeoutError):
                    msg = "The AI model took too long to respond. Please try again."
                else:
                    msg = f"Inference failed ({err_name}). Please try again."
                logger.error(f"Inference failed after retries: {last_inference_error}")
                await _emit(run, {"type": "error", "content": msg})
                return

            text_content = assistant_msg.content
            tool_calls = assistant_msg.tool_calls

            tc_for_db = None
            if tool_calls:
                total_tool_calls += len(tool_calls)
                tc_for_db = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            await db.add_message(
                chat_id, "assistant", text_content, tool_calls=tc_for_db
            )

            assistant_dict: dict = {"role": "assistant"}
            if text_content:
                assistant_dict["content"] = text_content
            if tc_for_db:
                assistant_dict["tool_calls"] = tc_for_db
            messages.append(assistant_dict)

            if text_content:
                await _emit(run, {"type": "text", "content": text_content})

            if not tool_calls:
                # Auto-skill nudge for SSE path
                skill_text = await _maybe_skill_nudge(
                    total_tool_calls, messages, chat_id,
                    store_history=True, tools=tools,
                )
                if skill_text:
                    await _emit(run, {"type": "text", "content": skill_text})
                return

            # Emit tool_use SSE events for all tools before execution
            for tc in tool_calls:
                await _emit(run, {"type": "tool_use", "name": tc.function.name, "input": tc.function.arguments})

            # Execute tool calls concurrently
            async def _exec_tool_sse(tc):
                tool_name = tc.function.name
                arguments = tc.function.arguments

                # Plugin pre_tool hook: may modify arguments
                if _plugin_manager is not None:
                    arguments = await _plugin_manager.fire_modify(
                        "pre_tool", arguments, tool_name,
                    )

                if tool_name == "spawn":
                    result = await _handle_spawn(arguments, chat_id)
                else:
                    result = await execute_tool(
                        tool_name, arguments,
                        image_callback=_stash_images,
                    )

                # Plugin post_tool hook: may modify result
                if _plugin_manager is not None:
                    result = await _plugin_manager.fire_modify(
                        "post_tool", result, tool_name,
                    )

                return result

            results = await asyncio.gather(
                *[_exec_tool_sse(tc) for tc in tool_calls], return_exceptions=True
            )

            # Process results in original order
            for tc, result in zip(tool_calls, results):
                if isinstance(result, BaseException):
                    logger.error(
                        f"Tool {tc.function.name} raised: {result}", exc_info=result
                    )
                    result = f"Error executing {tc.function.name}: {result}"

                # Detect send_file markers and emit file SSE event
                if isinstance(result, str) and result.startswith("__SEND_FILE__:"):
                    parts = result.split(":", 2)
                    rel_path = parts[1] if len(parts) > 1 else ""
                    caption = parts[2] if len(parts) > 2 else ""
                    await _emit(run, {"type": "file", "path": rel_path, "caption": caption})
                    result = f"File sent to user: {rel_path}"

                await db.add_message(
                    chat_id, "tool", result, tool_call_id=tc.id
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Inject any images collected from tool results as a user message
            if pending_images:
                messages.append({"role": "user", "content": list(pending_images)})
                pending_images.clear()

        await _emit(run, {"type": "text", "content": "(Reached maximum tool iterations)"})

    except asyncio.CancelledError:
        logger.info(f"Chat run cancelled for {chat_id}")
        # Force-append without await — we may be in a cancelled state
        run.events.append({"type": "error", "content": "Chat run was cancelled."})
    except Exception as e:
        logger.error(f"Chat run error for {chat_id}: {e}", exc_info=True)
        try:
            await _emit(run, {"type": "error", "content": str(e)})
        except asyncio.CancelledError:
            run.events.append({"type": "error", "content": str(e)})
    finally:
        # Fire session end hook (best-effort)
        if _plugin_manager is not None:
            try:
                await _plugin_manager.fire("on_session_end", chat_id)
            except Exception:
                pass  # never let plugin errors block cleanup
        # Always emit done and mark run as finished — force-append to avoid
        # re-cancellation in the finally block leaving run.done=False forever
        run.events.append({"type": "done"})
        run.done = True
        run.completed_at = time.time()
        try:
            async with run.condition:
                run.condition.notify_all()
        except asyncio.CancelledError:
            pass  # readers will see done=True on their next condition check


async def _read_run_events(run: ChatRun):
    """Async generator that reads from a ChatRun's event buffer.

    Yields SSE-formatted strings. Sends keepalives if no new events arrive
    within _KEEPALIVE_INTERVAL. When the client disconnects the generator
    is cancelled but the background task continues unaffected.
    """
    cursor = 0
    while True:
        # Wait for new events, done signal, or keepalive timeout
        deadline = asyncio.get_running_loop().time() + _KEEPALIVE_INTERVAL
        async with run.condition:
            while cursor >= len(run.events) and not run.done:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        run.condition.wait(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break

        # Yield any buffered events from cursor onward
        while cursor < len(run.events):
            yield _sse_event(run.events[cursor])
            cursor += 1

        # If the run is done and we've consumed everything, exit
        if run.done and cursor >= len(run.events):
            return

        # No new events within the keepalive interval — send keepalive
        if cursor >= len(run.events) and not run.done:
            yield _sse_keepalive()


def cancel_chat_run(chat_id: str) -> bool:
    """Cancel an active chat run. Returns True if something was cancelled."""
    existing = _active_runs.get(chat_id)
    if existing and not existing.done and not existing.task.done():
        existing.task.cancel()
        return True
    return False


@app.post("/chat")
async def chat(req: ChatRequest):
    """Handle a proxied chat message with SSE streaming and tool use."""
    _prune_old_chat_runs()

    # Cancel any existing active run for this chat.
    # The background task's finally block will set done=True and notify readers.
    existing = _active_runs.get(req.chat_id)
    if existing and not existing.done and not existing.task.done():
        existing.task.cancel()

    # Create a new ChatRun
    run = ChatRun(
        chat_id=req.chat_id,
        task=None,  # type: ignore — will be set immediately below
        user_message=req.message,
    )
    run.task = asyncio.create_task(_run_chat_background(run, req.chat_id, req.message))
    _active_runs[req.chat_id] = run

    return StreamingResponse(_read_run_events(run), media_type="text/event-stream")


@app.get("/chat/{chat_id}/active")
async def chat_active(chat_id: str):
    """Check if there is an active (in-progress) chat run for this chat_id."""
    run = _active_runs.get(chat_id)
    if run and not run.done:
        return {"active": True, "user_message": run.user_message}
    return {"active": False, "user_message": ""}


@app.get("/chat/{chat_id}/stream")
async def chat_stream(chat_id: str):
    """Reconnect to an active chat run's SSE stream.

    If a run is still in progress, replays all buffered events then continues
    streaming live. First event is a stream_meta with the original user_message.
    If no active run exists, returns a single done event.
    """
    run = _active_runs.get(chat_id)
    if run is None or run.done:
        # No active run — just return done so the client knows
        async def _done_stream():
            yield _sse_event({"type": "done"})
        return StreamingResponse(_done_stream(), media_type="text/event-stream")

    async def _reconnect_stream():
        # Send metadata so the frontend knows which message is being processed
        yield _sse_event({"type": "stream_meta", "user_message": run.user_message})
        async for event in _read_run_events(run):
            yield event

    return StreamingResponse(_reconnect_stream(), media_type="text/event-stream")


def _created_at_to_ms(s: str | None) -> int | None:
    """Parse the DB's `datetime('now')` text (UTC, 'YYYY-MM-DD HH:MM:SS')
    into a Unix ms timestamp. Returns None if unparseable."""
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str, limit: int = 50):
    """Return conversation history as ChatMessage events for the frontend."""
    messages = await db.get_history(chat_id, limit=limit, include_timestamps=True)
    events = []
    for msg in messages:
        role = msg["role"]
        ts_ms = _created_at_to_ms(msg.get("created_at"))
        ts_field = {"timestamp_ms": ts_ms} if ts_ms is not None else {}
        if role == "user":
            events.append({"type": "text", "content": msg.get("content", ""), "name": "user", **ts_field})
        elif role == "assistant":
            if msg.get("content"):
                events.append({"type": "text", "content": msg["content"], **ts_field})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    events.append({
                        "type": "tool_use",
                        "name": fn.get("name", ""),
                        "input": fn.get("arguments", ""),
                        **ts_field,
                    })
        elif role == "tool":
            # Reconstruct file events from stored tool results
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("File sent to user: "):
                rel_path = content.removeprefix("File sent to user: ").strip()
                if rel_path:
                    events.append({"type": "file", "path": rel_path, **ts_field})
    return {"messages": events}


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Clear conversation history for a chat."""
    count = await db.clear_history(chat_id)
    return {"status": "ok", "deleted": count}


@app.get("/pending")
async def get_pending():
    """Return pending proactive messages and clear them."""
    messages = await db.get_and_clear_pending()
    return {"messages": messages}


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve a workspace file (protected by auth middleware)."""
    try:
        resolved = validate_workspace_path(
            file_path, settings.workspace_path, must_exist=True, reject_sensitive=True
        )
        return FileResponse(resolved, filename=resolved.name)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": str(e)})


# ── Direct download tokens ──────────────────────────────────────────
# Short-lived HMAC tokens that let users download files directly from
# the VM without proxying through the LiberClaw API.

_DOWNLOAD_TOKEN_TTL = 300  # 5 minutes


def _sign_download_token(file_path: str, expires: int) -> str:
    """Create an HMAC-SHA256 signature for a download token."""
    msg = f"{file_path}\n{expires}".encode()
    return _hmac.new(
        settings.agent_secret_hash.encode(), msg, hashlib.sha256
    ).hexdigest()


@app.post("/files/token")
async def create_download_token(request: Request):
    """Generate a short-lived download token for a file (requires auth)."""
    body = await request.json()
    file_path = body.get("path", "")
    if not file_path:
        return JSONResponse(status_code=400, content={"error": "path is required"})
    # Validate the file exists and is safe to serve
    try:
        validate_workspace_path(
            file_path, settings.workspace_path, must_exist=True, reject_sensitive=True
        )
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": str(e)})
    expires = int(time.time()) + _DOWNLOAD_TOKEN_TTL
    sig = _sign_download_token(file_path, expires)
    token = f"{file_path}:{expires}:{sig}"
    url_token = base64.urlsafe_b64encode(token.encode()).decode()
    return {"token": url_token, "expires_in": _DOWNLOAD_TOKEN_TTL}


@app.get("/dl/{token}")
async def direct_download(token: str):
    """Serve a file using a short-lived signed token (no auth required)."""
    try:
        decoded = base64.urlsafe_b64decode(token).decode()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid token"})
    parts = decoded.rsplit(":", 2)
    if len(parts) != 3:
        return JSONResponse(status_code=400, content={"error": "invalid token"})
    file_path, expires_str, sig = parts
    try:
        expires = int(expires_str)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "invalid token"})
    if time.time() > expires:
        return JSONResponse(status_code=410, content={"error": "token expired"})
    expected_sig = _sign_download_token(file_path, expires)
    if not _hmac.compare_digest(sig, expected_sig):
        return JSONResponse(status_code=403, content={"error": "invalid signature"})
    try:
        resolved = validate_workspace_path(
            file_path, settings.workspace_path, must_exist=True, reject_sensitive=True
        )
        # Sanitize filename for Content-Disposition header
        import re as _re
        safe_name = _re.sub(r'[^\w\-. ]', '', resolved.name).strip() or "download"
        return FileResponse(resolved, filename=safe_name)
    except PathSecurityError:
        return JSONResponse(status_code=403, content={"error": "access denied"})


@app.get("/workspace/tree")
async def workspace_tree(max_depth: int = 5):
    """Return recursive workspace file tree."""
    SKIP_NAMES = {".env", ".git", "agent.db", "agent.db-shm", "agent.db-wal", "__pycache__", "node_modules"}

    def walk(path: Path, depth: int) -> list[dict]:
        if depth <= 0 or not path.is_dir():
            return []
        entries = []
        try:
            for entry in sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name in SKIP_NAMES:
                    continue
                rel = str(entry.relative_to(workspace_root))
                if entry.is_dir():
                    entries.append({"name": entry.name, "path": rel, "type": "dir", "children": walk(entry, depth - 1)})
                else:
                    entries.append({"name": entry.name, "path": rel, "type": "file", "size": entry.stat().st_size})
        except PermissionError:
            pass
        return entries

    workspace_root = Path(settings.workspace_path).resolve()
    return {"tree": walk(workspace_root, max_depth)}


@app.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    path: str = Form(default="uploads"),
):
    """Upload a file to the agent workspace."""
    workspace_root = Path(settings.workspace_path).resolve()
    try:
        target_dir = validate_workspace_path(path, settings.workspace_path)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": str(e)})

    target_dir.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "filename is required"})
    safe_filename = Path(file.filename).name
    target_file = target_dir / safe_filename

    content = await file.read()
    if len(content) > MAX_SEND_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large (50MB max)"})

    # Validate final path is within workspace
    try:
        target_file.resolve().relative_to(workspace_root)
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "Invalid filename"})

    target_file.write_bytes(content)
    rel = str(target_file.relative_to(workspace_root))
    return {"path": rel, "size": len(content), "name": safe_filename}


@app.get("/health")
async def health():
    from baal_agent import AGENT_VERSION

    return {"status": "ok", "agent_name": settings.agent_name, "version": AGENT_VERSION, "capabilities": ["vision"]}


@app.get("/info")
async def info():
    return {
        "agent_name": settings.agent_name,
        "model": settings.model,
        "has_system_prompt": bool(settings.system_prompt),
        "capabilities": ["vision"],
        "tools": [t["function"]["name"] for t in get_tool_definitions(include_spawn=True)],
        "heartbeat_interval": settings.heartbeat_interval,
    }


# ── Subagent management endpoints ─────────────────────────────────────

@app.get("/subagents")
async def list_subagents():
    """List all subagent runs, running first, then by started_at descending."""
    _prune_old_subagent_runs()
    runs = sorted(
        _subagent_runs.values(),
        key=lambda r: (r.status != "running", -r.started_at),
    )
    return {
        "subagents": [
            {
                "id": r.id,
                "label": r.label,
                "task": r.task[:200],
                "status": r.status,
                "chat_id": r.chat_id,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "result_preview": (r.result or "")[:200] if r.result else None,
                "error": r.error,
                "duration": (
                    (r.completed_at or time.time()) - r.started_at
                ),
            }
            for r in runs
        ]
    }


@app.get("/subagents/{run_id}")
async def get_subagent(run_id: str):
    """Get full details of a single subagent run."""
    run = _subagent_runs.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": f"Subagent run '{run_id}' not found"})
    return {
        "id": run.id,
        "label": run.label,
        "task": run.task,
        "persona": run.persona,
        "status": run.status,
        "chat_id": run.chat_id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "result": run.result,
        "error": run.error,
        "duration": (run.completed_at or time.time()) - run.started_at,
    }


@app.post("/subagents/{run_id}/stop")
async def stop_subagent(run_id: str):
    """Cancel a running subagent."""
    run = _subagent_runs.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": f"Subagent run '{run_id}' not found"})
    if run.status != "running":
        return JSONResponse(status_code=400, content={"error": f"Subagent '{run_id}' is not running (status: {run.status})"})

    # Cancel the asyncio task
    if run.asyncio_task and not run.asyncio_task.done():
        run.asyncio_task.cancel()

    run.status = "failed"
    run.error = "Cancelled by user"
    run.completed_at = time.time()

    # Emit failed event
    await db.add_pending(
        run.chat_id,
        json.dumps({"type": "subagent_failed", "run_id": run.id, "label": run.label, "status": "failed", "error": run.error}),
        source="subagent_event",
    )

    return {"status": "ok", "run_id": run_id, "message": f"Subagent '{run.label}' cancelled"}


# ── Telegram management endpoints ────────────────────────────────────


@app.get("/telegram/status")
async def telegram_status():
    """Return Telegram bot connection status."""
    if _telegram_bot and _telegram_bot.connected:
        return {
            "connected": True,
            "bot_username": _telegram_bot.bot_username,
            "bot_name": _telegram_bot.bot_name,
        }
    return {"connected": False, "bot_username": "", "bot_name": ""}


@app.get("/telegram/contacts")
async def telegram_contacts(status: str | None = None):
    """List Telegram contacts, optionally filtered by status."""
    contacts = await db.list_telegram_contacts(status=status)
    return {"contacts": contacts}


class TelegramContactUpdate(BaseModel):
    status: str


@app.patch("/telegram/contacts/{telegram_id}")
async def update_telegram_contact(telegram_id: str, body: TelegramContactUpdate):
    """Update a Telegram contact's status."""
    if body.status not in ("allowed", "pending", "blocked"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid status: {body.status}. Must be allowed, pending, or blocked."},
        )
    updated = await db.update_telegram_contact_status(telegram_id, body.status)
    if not updated:
        return JSONResponse(status_code=404, content={"error": "Contact not found"})
    return {"status": "ok", "telegram_id": telegram_id, "new_status": body.status}


@app.delete("/telegram/contacts/{telegram_id}")
async def delete_telegram_contact(telegram_id: str):
    """Remove a Telegram contact."""
    deleted = await db.delete_telegram_contact(telegram_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Contact not found"})
    return {"status": "ok", "telegram_id": telegram_id}


# ── Local web UI static mount (must be last, after all API routes) ─────

if settings.local_ui_enabled:
    _dist_dir = (
        Path(settings.local_ui_dist_path)
        if settings.local_ui_dist_path
        else Path(__file__).parent / "webui" / "dist"
    )
    if _dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="webui")
