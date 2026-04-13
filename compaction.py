"""Token-aware context management with history compaction via summarization."""

from __future__ import annotations

import json
import logging

from baal_agent.config import AgentSettings
from baal_agent.database import AgentDatabase
from baal_agent.image_utils import strip_images_from_content
from baal_agent.inference import InferenceClient

logger = logging.getLogger(__name__)

# Known model context window sizes (tokens).
_MODEL_CONTEXT_SIZES = {
    "qwen3-coder-next": 200_000,
    "glm-4.7": 200_000,
    "hermes-3-8b-tee": 200_000,
    "claw-flash": 200_000,
    "claw-core": 200_000,
    "deep-claw": 200_000,
}
_DEFAULT_CONTEXT_SIZE = 200_000


def get_context_limit(model: str, configured_max: int) -> int:
    """Return the context token limit for the given model."""
    if configured_max > 0:
        return configured_max
    return _MODEL_CONTEXT_SIZES.get(model, _DEFAULT_CONTEXT_SIZE)


_IMAGE_TOKEN_ESTIMATE = 1000


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for a list of chat messages.

    Uses chars/2 heuristic plus per-message overhead.  Conservative ratio
    because code, JSON, tool call IDs, and structured data tokenize at
    significantly worse than chars/4 with BPE tokenizers.

    Multimodal content (list of blocks) is handled by summing text block
    lengths and adding _IMAGE_TOKEN_ESTIMATE per image block.
    """
    total_chars = 0
    image_count = 0
    for msg in messages:
        content = msg.get("content")
        if content:
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        total_chars += len(block["text"])
                    elif block.get("type") == "image_url":
                        image_count += 1
            else:
                total_chars += len(content)
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))
        if msg.get("tool_call_id"):
            total_chars += len(msg["tool_call_id"])
    return total_chars // 2 + image_count * _IMAGE_TOKEN_ESTIMATE + 4 * len(messages)


_COMPACTION_PROMPT = (
    "Summarize the conversation above into these sections:\n"
    "## Key Facts\n"
    "Specific details: names, URLs, numbers, file paths, technical decisions.\n"
    "## Decisions Made\n"
    "What was decided and why.\n"
    "## User Preferences\n"
    "Communication style, preferences, constraints mentioned by the user.\n"
    "## Active Tasks\n"
    "Ongoing work, next steps, unfinished items.\n"
    "## Important Context\n"
    "Anything else needed to continue the conversation.\n\n"
    "Be thorough but concise. Preserve specific details over generalities."
)

_COMPACTION_UPDATE_PROMPT = (
    "The conversation above continues from a previous summary. "
    "Update the existing summary with new information from the conversation. "
    "Keep the same section structure (Key Facts, Decisions Made, User Preferences, "
    "Active Tasks, Important Context). Add new items, update changed items, "
    "mark completed tasks. Preserve all sections even if unchanged."
)

_MEMORY_FLUSH_PROMPT = (
    "The conversation context is about to be compressed. Review the recent messages "
    "and save any important information to your memory files that might be lost. "
    "Focus on: user preferences, important decisions, URLs, specific facts, and "
    "ongoing task details. Write to workspace/memory/MEMORY.md or today's daily note. "
    "If nothing important needs saving, do nothing."
)

# Memory-only tools for the pre-compaction flush
MEMORY_FLUSH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace."},
                    "old_string": {"type": "string", "description": "Text to find."},
                    "new_string": {"type": "string", "description": "Text to replace with."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]


async def _memory_flush(
    inference: InferenceClient,
    model: str,
    system_prompt: str,
    recent_messages: list[dict],
    settings: AgentSettings,
) -> None:
    """Run a mini agent turn to persist important info before compaction.

    Uses a restricted tool set (read_file, write_file, edit_file) and up to
    3 iterations so the agent can read and update memory files.
    """
    from baal_agent.tools import execute_tool

    try:
        flush_messages: list[dict] = [{"role": "system", "content": system_prompt}]
        flush_messages.extend(recent_messages)
        flush_messages.append({"role": "user", "content": _MEMORY_FLUSH_PROMPT})

        for _iteration in range(3):
            response = await inference.chat(
                flush_messages, model=model, tools=MEMORY_FLUSH_TOOLS
            )

            tool_calls = response.tool_calls
            if not tool_calls:
                break  # agent decided nothing needs saving

            # Record assistant message in the flush conversation
            assistant_dict: dict = {"role": "assistant"}
            if response.content:
                assistant_dict["content"] = response.content
            if tool_calls:
                assistant_dict["tool_calls"] = [
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
            flush_messages.append(assistant_dict)

            for tc in tool_calls:
                result = await execute_tool(tc.function.name, tc.function.arguments)
                flush_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        logger.info("Pre-compaction memory flush completed")
    except Exception as e:
        logger.warning(f"Pre-compaction memory flush failed (continuing): {e}")


def _inject_dynamic_context(messages: list[dict], dynamic_context: str) -> list[dict]:
    """Insert dynamic context (memory/skills) just before the last user message.

    This keeps the prefix [system(static) + history] stable across turns
    so the llama.cpp KV cache is preserved.
    """
    if not dynamic_context:
        return messages

    ctx_msg = {"role": "user", "content": f"[Context update]\n\n{dynamic_context}"}

    # Find the last user message and insert before it
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages.insert(i, ctx_msg)
            return messages

    # No user message found — append at end
    messages.append(ctx_msg)
    return messages


async def maybe_compact(
    db: AgentDatabase,
    inference: InferenceClient,
    chat_id: str,
    system_prompt: str,
    model: str,
    settings: AgentSettings,
    dynamic_context: str = "",
) -> list[dict]:
    """Build a messages list, compacting history if it exceeds the token budget.

    Returns a ready-to-use messages list with dynamic context (memory/skills)
    injected near the end for KV cache preservation.

    If the history exceeds the compaction threshold, older messages are
    summarized and replaced with a compact summary pair in the DB.
    """
    history = await db.get_history(chat_id, limit=settings.max_history)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    budget = get_context_limit(model, settings.max_context_tokens) - settings.generation_reserve
    trigger = int(budget * settings.compaction_threshold)
    tokens = estimate_tokens(messages)

    if tokens <= trigger:
        return _inject_dynamic_context(messages, dynamic_context)

    logger.info(
        f"Context for {chat_id} exceeds threshold ({tokens} > {trigger} tokens, "
        f"budget={budget}), compacting"
    )

    # Dynamic keep: scale down under extreme context pressure (Task 1.3)
    # Gradient from keep_max (at trigger) to keep_min (at budget).
    # pressure=0 at trigger, pressure=1 at budget.
    keep_max = settings.compaction_keep_messages
    pressure = min(1.0, (tokens - trigger) / max(1, budget - trigger))
    keep = int(keep_max - pressure * (keep_max - settings.compaction_keep_min))
    keep = max(settings.compaction_keep_min, min(keep, keep_max))
    if keep < keep_max:
        logger.info(f"Context pressure: keeping {keep} messages (max={keep_max})")

    if len(history) <= keep:
        # Even recent-only exceeds budget — just return what we have,
        # the model will do its best with truncated input.
        return _inject_dynamic_context(messages, dynamic_context)

    old = history[:-keep]
    recent = history[-keep:]

    # Pre-compaction memory flush (Task 1.1): persist important info before
    # old messages are summarized away.
    if settings.compaction_flush_enabled:
        await _memory_flush(inference, model, system_prompt, recent, settings)

    # Detect if the first old message is already a summary (Task 1.2).
    # If so, use the update prompt to refine the existing summary rather than
    # creating one from scratch.
    is_update = (
        len(old) >= 2
        and old[0].get("role") == "user"
        and isinstance(old[0].get("content"), str)
        and "[Earlier conversation summary]" in old[0]["content"]
    )
    prompt = _COMPACTION_UPDATE_PROMPT if is_update else _COMPACTION_PROMPT

    # Strip images from old messages before sending to the compaction LLM.
    stripped_old = []
    for msg in old:
        content = msg.get("content")
        if isinstance(content, list):
            stripped_msg = {**msg, "content": strip_images_from_content(content)}
            stripped_old.append(stripped_msg)
        else:
            stripped_old.append(msg)

    # Build compaction request reusing the system prompt prefix for cache hits.
    compaction_messages = [{"role": "system", "content": system_prompt}]
    compaction_messages.extend(stripped_old)
    compaction_messages.append({"role": "user", "content": prompt})

    # If the compaction request itself exceeds the context budget, iteratively
    # halve the old messages until it fits (oldest are dropped).
    compaction_tokens = estimate_tokens(compaction_messages)
    while compaction_tokens > budget and len(stripped_old) > 2:
        dropped = len(stripped_old) // 2
        old = old[dropped:]
        stripped_old = stripped_old[dropped:]
        compaction_messages = [{"role": "system", "content": system_prompt}]
        compaction_messages.extend(stripped_old)
        compaction_messages.append({"role": "user", "content": prompt})
        compaction_tokens = estimate_tokens(compaction_messages)
        logger.info(f"Compaction request too large, dropped {dropped} oldest messages")

    try:
        summary_msg = await inference.chat(
            compaction_messages, model=model, tools=None
        )
        summary = summary_msg.content or "(no summary generated)"
    except Exception as e:
        logger.error(f"Compaction inference failed: {e}")
        # Fall back to un-compacted messages rather than losing the conversation
        return _inject_dynamic_context(messages, dynamic_context)

    await db.compact_history(
        chat_id, keep_recent=keep, summary=summary
    )

    # Reload from DB to get the clean state
    history = await db.get_history(chat_id, limit=settings.max_history)
    result = [{"role": "system", "content": system_prompt}]
    result.extend(history)

    new_tokens = estimate_tokens(result)
    logger.info(
        f"Compaction complete for {chat_id}: {tokens} -> {new_tokens} tokens "
        f"({len(old)} old messages summarized, {len(recent)} kept)"
    )
    return _inject_dynamic_context(result, dynamic_context)
