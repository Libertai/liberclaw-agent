"""FastAPI application deployed to each agent VM."""

import asyncio
import hashlib
import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel

from baal_agent.compaction import maybe_compact
from baal_agent.config import AgentSettings
from baal_agent.context import build_subagent_prompt, build_system_prompt
from baal_agent.database import AgentDatabase
from baal_agent.inference import InferenceClient
from baal_agent.security import MAX_SEND_FILE_SIZE, PathSecurityError, validate_workspace_path
from baal_agent.tools import configure_tools, execute_tool, get_tool_definitions

logger = logging.getLogger(__name__)

settings = AgentSettings()
db = AgentDatabase(db_path=settings.db_path)
inference = InferenceClient(api_key=settings.libertai_api_key)

_heartbeat_task: asyncio.Task | None = None


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


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _heartbeat_task
    await db.initialize()
    configure_tools(settings.workspace_path)

    # Ensure workspace directories exist
    workspace = Path(settings.workspace_path)
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "skills").mkdir(parents=True, exist_ok=True)

    # Start heartbeat if configured
    if settings.heartbeat_interval > 0:
        _heartbeat_task = asyncio.create_task(_heartbeat_loop())

    yield

    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
    await db.close()


app = FastAPI(title=f"Baal Agent: {settings.agent_name}", lifespan=lifespan)


# ── Auth middleware ────────────────────────────────────────────────────

@app.middleware("http")
async def verify_auth(request: Request, call_next):
    """Reject requests without a valid Bearer token (except /health)."""
    if request.url.path == "/health":
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    token_hash = hashlib.sha256(token.encode()).hexdigest() if token else ""
    if not token or not secrets.compare_digest(token_hash, settings.agent_secret_hash):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


# ── Core agentic loop ─────────────────────────────────────────────────

async def _run_agent_turn(
    message: str,
    chat_id: str,
    *,
    restricted: bool = False,
    max_iterations: int | None = None,
    store_history: bool = True,
    file_events: list[dict] | None = None,
    system_prompt_override: str | None = None,
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

    Returns:
        The final text response, or None if no text was generated.
    """
    iterations = max_iterations or settings.max_tool_iterations
    tools = get_tool_definitions(include_spawn=not restricted)
    tool_names = [t["function"]["name"] for t in tools]

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
        )

    if store_history:
        await db.add_message(chat_id, "user", message)
        messages = await maybe_compact(
            db, inference, chat_id, system_prompt, settings.model, settings
        )
    else:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": message})

    final_text = None

    for _iteration in range(iterations):
        assistant_msg = await inference.chat(
            messages=messages, model=settings.model, tools=tools
        )

        text_content = assistant_msg.content
        tool_calls = assistant_msg.tool_calls

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
            return final_text

        for tc in tool_calls:
            name = tc.function.name
            arguments = tc.function.arguments

            # Handle spawn tool specially
            if name == "spawn" and not restricted:
                result = await _handle_spawn(arguments, chat_id)
            else:
                result = await execute_tool(name, arguments)

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

    return final_text


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
        # Forward file events
        for fe in files:
            await db.add_pending(
                origin_chat_id,
                json.dumps({"type": "file", "path": fe["path"], "caption": fe["caption"]}),
                source="subagent_file",
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


# ── Heartbeat ─────────────────────────────────────────────────────────

def _is_heartbeat_empty(content: str) -> bool:
    """Check if heartbeat file has no actionable content."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        # Unchecked checkbox counts as actionable
        if stripped.startswith("- [ ]"):
            return False
        # Any non-header, non-comment text is actionable
        return False
    return True


async def _heartbeat_loop():
    """Periodic heartbeat — check HEARTBEAT.md and run tasks."""
    while True:
        await asyncio.sleep(settings.heartbeat_interval)
        try:
            heartbeat_file = Path(settings.workspace_path) / "HEARTBEAT.md"
            if not heartbeat_file.exists():
                continue
            content = heartbeat_file.read_text()
            if _is_heartbeat_empty(content):
                continue

            files: list[dict] = []
            result = await _run_agent_turn(
                "Read HEARTBEAT.md and follow any instructions or tasks listed there. "
                "If nothing needs attention, reply with just: HEARTBEAT_OK",
                chat_id="__heartbeat__",
                store_history=False,
                file_events=files,
            )

            if result and "HEARTBEAT_OK" not in result.upper().replace("_", ""):
                if settings.owner_chat_id:
                    await db.add_pending(
                        settings.owner_chat_id,
                        f"[Heartbeat] {result}",
                        source="heartbeat",
                    )
            if settings.owner_chat_id:
                for fe in files:
                    await db.add_pending(
                        settings.owner_chat_id,
                        json.dumps({"type": "file", "path": fe["path"], "caption": fe["caption"]}),
                        source="heartbeat_file",
                    )
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


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


async def _with_keepalive(coro, queue: asyncio.Queue):
    """Run a coroutine while periodically pushing keepalive signals to a queue.

    The queue is used to interleave keepalive comments with SSE data events
    in the event_stream generator.  When the coroutine completes, the result
    is placed on the queue as a ("result", value) tuple.  Exceptions are
    placed as ("error", exc).
    """
    task = asyncio.create_task(coro)
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_KEEPALIVE_INTERVAL)
        except asyncio.TimeoutError:
            # Task still running — send keepalive
            await queue.put(("keepalive", None))
        except Exception:
            # Task raised — will be handled below
            break

    try:
        result = task.result()
        await queue.put(("result", result))
    except Exception as e:
        await queue.put(("error", e))


# ── Endpoints ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    chat_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Handle a proxied chat message with SSE streaming and tool use."""

    async def event_stream():
        try:
            tools = get_tool_definitions(include_spawn=True)
            tool_names = [t["function"]["name"] for t in tools]

            system_prompt = build_system_prompt(
                settings.system_prompt,
                settings.agent_name,
                settings.workspace_path,
                tool_names=tool_names,
            )

            await db.add_message(req.chat_id, "user", req.message)
            messages = await maybe_compact(
                db, inference, req.chat_id, system_prompt, settings.model, settings
            )

            inference_timeout = settings.inference_timeout

            for _iteration in range(settings.max_tool_iterations):
                # Call inference with keepalive + loop-level retry.
                assistant_msg = None
                last_inference_error = None

                for _inf_attempt in range(_LOOP_INFERENCE_RETRIES + 1):
                    inference_coro = inference.chat(
                        messages=messages, model=settings.model, tools=tools
                    )
                    inference_q: asyncio.Queue = asyncio.Queue()
                    inference_keepalive = asyncio.create_task(
                        _with_keepalive(inference_coro, inference_q)
                    )
                    try:
                        while True:
                            msg_type, msg_val = await asyncio.wait_for(
                                inference_q.get(), timeout=inference_timeout
                            )
                            if msg_type == "keepalive":
                                yield _sse_keepalive()
                            elif msg_type == "result":
                                assistant_msg = msg_val
                                break
                            elif msg_type == "error":
                                raise msg_val
                        break  # success — exit retry loop
                    except Exception as e:
                        last_inference_error = e
                        if _is_retryable(e) and _inf_attempt < _LOOP_INFERENCE_RETRIES:
                            logger.warning(
                                f"Inference attempt {_inf_attempt + 1} failed "
                                f"({type(e).__name__}), retrying in {_LOOP_RETRY_DELAY}s"
                            )
                            yield _sse_event({
                                "type": "text",
                                "content": "\n\n*[Inference error, retrying...]*\n\n",
                            })
                            await asyncio.sleep(_LOOP_RETRY_DELAY)
                            continue
                        else:
                            break  # non-retryable or exhausted retries
                    finally:
                        inference_keepalive.cancel()
                        try:
                            await inference_keepalive
                        except asyncio.CancelledError:
                            pass

                if assistant_msg is None:
                    err_name = type(last_inference_error).__name__ if last_inference_error else "Unknown"
                    if isinstance(last_inference_error, asyncio.TimeoutError):
                        msg = "The AI model took too long to respond. Please try again."
                    else:
                        msg = f"Inference failed ({err_name}). Please try again."
                    logger.error(f"Inference failed after retries: {last_inference_error}")
                    yield _sse_event({"type": "error", "content": msg})
                    yield _sse_event({"type": "done"})
                    return

                text_content = assistant_msg.content
                tool_calls = assistant_msg.tool_calls

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

                await db.add_message(
                    req.chat_id, "assistant", text_content, tool_calls=tc_for_db
                )

                assistant_dict: dict = {"role": "assistant"}
                if text_content:
                    assistant_dict["content"] = text_content
                if tc_for_db:
                    assistant_dict["tool_calls"] = tc_for_db
                messages.append(assistant_dict)

                if text_content:
                    yield _sse_event({"type": "text", "content": text_content})

                if not tool_calls:
                    yield _sse_event({"type": "done"})
                    return

                for tc in tool_calls:
                    name = tc.function.name
                    arguments = tc.function.arguments
                    yield _sse_event({"type": "tool_use", "name": name, "input": arguments})

                    # Execute tool with keepalive to prevent proxy timeouts
                    # during long-running operations (e.g., bash commands).
                    keepalive_q: asyncio.Queue = asyncio.Queue()
                    if name == "spawn":
                        coro = _handle_spawn(arguments, req.chat_id)
                    else:
                        coro = execute_tool(name, arguments)

                    keepalive_task = asyncio.create_task(
                        _with_keepalive(coro, keepalive_q)
                    )

                    result = None
                    tool_error = None
                    while True:
                        msg_type, msg_val = await keepalive_q.get()
                        if msg_type == "keepalive":
                            yield _sse_keepalive()
                        elif msg_type == "result":
                            result = msg_val
                            break
                        elif msg_type == "error":
                            tool_error = msg_val
                            break

                    # Ensure the keepalive task is fully done
                    await keepalive_task

                    if tool_error is not None:
                        logger.error(
                            f"Tool {name} raised: {tool_error}", exc_info=tool_error
                        )
                        result = f"Error executing {name}: {tool_error}"

                    # Detect send_file markers and emit file SSE event
                    if isinstance(result, str) and result.startswith("__SEND_FILE__:"):
                        parts = result.split(":", 2)
                        rel_path = parts[1] if len(parts) > 1 else ""
                        caption = parts[2] if len(parts) > 2 else ""
                        yield _sse_event({"type": "file", "path": rel_path, "caption": caption})
                        result = f"File sent to user: {rel_path}"

                    await db.add_message(
                        req.chat_id, "tool", result, tool_call_id=tc.id
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            yield _sse_event({"type": "text", "content": "(Reached maximum tool iterations)"})
            yield _sse_event({"type": "done"})

        except asyncio.CancelledError:
            # Client disconnected — log but don't try to yield (stream is dead)
            logger.info(f"Client disconnected during chat stream for {req.chat_id}")
            return
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            try:
                yield _sse_event({"type": "error", "content": str(e)})
                yield _sse_event({"type": "done"})
            except Exception:
                # If we can't even yield the error (broken pipe), just bail
                logger.warning("Failed to send error event to client")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str, limit: int = 50):
    """Return conversation history as ChatMessage events for the frontend."""
    messages = await db.get_history(chat_id, limit=limit)
    events = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            events.append({"type": "text", "content": msg.get("content", ""), "name": "user"})
        elif role == "assistant":
            if msg.get("content"):
                events.append({"type": "text", "content": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    events.append({
                        "type": "tool_use",
                        "name": fn.get("name", ""),
                        "input": fn.get("arguments", ""),
                    })
        elif role == "tool":
            # Reconstruct file events from stored tool results
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("File sent to user: "):
                rel_path = content.removeprefix("File sent to user: ").strip()
                if rel_path:
                    events.append({"type": "file", "path": rel_path})
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
    target_file = target_dir / file.filename

    content = await file.read()
    if len(content) > MAX_SEND_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large (50MB max)"})

    target_file.write_bytes(content)
    rel = str(target_file.relative_to(workspace_root))
    return {"path": rel, "size": len(content), "name": file.filename}


@app.get("/health")
async def health():
    from baal_agent import AGENT_VERSION

    return {"status": "ok", "agent_name": settings.agent_name, "version": AGENT_VERSION}


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
