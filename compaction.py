"""Token-aware context management with history compaction via summarization."""

from __future__ import annotations

import json
import logging

from baal_agent.config import AgentSettings
from baal_agent.database import AgentDatabase
from baal_agent.inference import InferenceClient

logger = logging.getLogger(__name__)

# Known model context window sizes (tokens).
_MODEL_CONTEXT_SIZES = {
    "qwen3-coder-next": 98_304,
    "glm-4.7": 131_072,
}
_DEFAULT_CONTEXT_SIZE = 32_768


def get_context_limit(model: str, configured_max: int) -> int:
    """Return the context token limit for the given model."""
    if configured_max > 0:
        return configured_max
    return _MODEL_CONTEXT_SIZES.get(model, _DEFAULT_CONTEXT_SIZE)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for a list of chat messages.

    Uses chars/4 heuristic plus per-message overhead.  Good enough for a
    budget guard — doesn't need to be billing-precise.
    """
    total_chars = 0
    for msg in messages:
        if msg.get("content"):
            total_chars += len(msg["content"])
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))
        if msg.get("tool_call_id"):
            total_chars += len(msg["tool_call_id"])
    return total_chars // 4 + 4 * len(messages)


_COMPACTION_PROMPT = (
    "Provide a concise summary of the conversation above. "
    "Capture key facts, decisions, user preferences, and any ongoing tasks. "
    "Be thorough but brief."
)


async def maybe_compact(
    db: AgentDatabase,
    inference: InferenceClient,
    chat_id: str,
    system_prompt: str,
    model: str,
    settings: AgentSettings,
) -> list[dict]:
    """Build a messages list, compacting history if it exceeds the token budget.

    Returns a ready-to-use messages list: [system_prompt] + history.
    If the history exceeds the available token budget, older messages are
    summarized and replaced with a compact summary pair in the DB.
    """
    history = await db.get_history(chat_id, limit=settings.max_history)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    budget = get_context_limit(model, settings.max_context_tokens) - settings.generation_reserve
    tokens = estimate_tokens(messages)

    if tokens <= budget:
        return messages

    logger.info(
        f"Context for {chat_id} exceeds budget ({tokens} > {budget} tokens), compacting"
    )

    keep = settings.compaction_keep_messages
    if len(history) <= keep:
        # Even recent-only exceeds budget — just return what we have,
        # the model will do its best with truncated input.
        return messages

    old = history[:-keep]
    recent = history[-keep:]

    # Build compaction request reusing the system prompt prefix for cache hits.
    compaction_messages = [{"role": "system", "content": system_prompt}]
    compaction_messages.extend(old)
    compaction_messages.append({"role": "user", "content": _COMPACTION_PROMPT})

    try:
        summary_msg = await inference.chat(
            compaction_messages, model=model, tools=None
        )
        summary = summary_msg.content or "(no summary generated)"
    except Exception as e:
        logger.error(f"Compaction inference failed: {e}")
        # Fall back to un-compacted messages rather than losing the conversation
        return messages

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
    return result
