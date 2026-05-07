"""Cross-platform messaging tool — send messages via connected platforms.

Provides a generic send_message tool that dispatches through registered
platform callbacks. Currently supports Telegram via the TelegramBot instance.
Extensible to Discord, Signal, Slack, etc. by registering additional callbacks.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

# Platform registry: maps platform name to a send function
# Signature: async def send_fn(chat_id: str, text: str) -> str
_platform_registry: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {}

# Known target registry for "list" action
_known_targets: dict[str, str] = {}  # name -> "platform:chat_id"


def register_platform(
    platform: str,
    send_fn: Callable[..., Coroutine[Any, Any, str]],
    known_targets: dict[str, str] | None = None,
) -> None:
    """Register a messaging platform with its send function.

    Args:
        platform: Platform name (e.g., 'telegram', 'discord').
        send_fn: Async callable(chat_id, text) -> result string.
        known_targets: Dict of {display_name: "platform:chat_id"} for the list action.
    """
    _platform_registry[platform] = send_fn
    if known_targets:
        _known_targets.update(known_targets)


def register_target(name: str, target: str) -> None:
    """Register a named target for discovery via the list action.

    Args:
        name: Human-readable name (e.g., 'Jon', '#engineering').
        target: Target string (e.g., 'telegram:123456', 'discord:999888777').
    """
    _known_targets[name] = target


SEND_MESSAGE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": (
            "Send a message to a connected messaging platform. "
            "Use action='list' first to see available targets. "
            "Target format: 'platform:chat_id' or a named target "
            "returned by list. Examples: 'telegram:1234567890', "
            "'discord:#general', 'discord:999888777:555444333'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list"],
                    "description": "'send' (default) dispatches a message. "
                    "'list' shows all available channels across connected platforms.",
                },
                "target": {
                    "type": "string",
                    "description": "Delivery target. Required for 'send'. "
                    "Format: 'platform' (uses home channel), "
                    "'platform:chat_id', or a named target. "
                    "For Telegram topics: 'telegram:chat_id:thread_id'.",
                },
                "message": {
                    "type": "string",
                    "description": "The message text to send. Required for 'send'.",
                },
            },
        },
    }
}


async def _exec_send_message(args: dict) -> str:
    """Execute the send_message tool."""
    action = (args.get("action") or "send").strip().lower()

    if action == "list":
        return _format_target_list()

    # Default to 'send' action
    target = (args.get("target") or "").strip()
    if not target:
        return "[error: 'target' is required for send]"

    message = (args.get("message") or "").strip()
    if not message:
        return "[error: 'message' is required for send]"

    dispatch_result = await _dispatch(target, message)
    return dispatch_result


def _format_target_list() -> str:
    """Format available targets into a readable list."""
    if not _known_targets:
        return "[info] No connected platforms available."

    lines = ["## Available Targets", ""]
    for name, target in sorted(_known_targets.items()):
        lines.append(f"- **{name}** → `{target}`")
    lines.append("")
    lines.append("Use target names directly in send_message(target='...').")
    return "\n".join(lines)


async def _dispatch(target: str, text: str) -> str:
    """Parse a target string and dispatch the message via the right platform."""
    # Check if target matches a known name
    if target in _known_targets:
        target = _known_targets[target]

    # Parse "platform:chat_id" or "platform:chat_id:thread_id"
    parts = target.split(":", 2)
    platform = parts[0].lower()
    chat_id = parts[1] if len(parts) > 1 else ""
    thread_id = parts[2] if len(parts) > 2 else None

    if not chat_id and platform == "local":
        return f"[ok] Message saved locally (no delivery target configured)"

    if not chat_id:
        return f"[error: missing chat_id in target '{target}'. Use format 'platform:chat_id']"

    send_fn = _platform_registry.get(platform)
    if not send_fn:
        platforms = ", ".join(_platform_registry.keys()) or "none"
        return (
            f"[error: No handler for platform '{platform}'. "
            f"Available: {platforms}]"
        )

    try:
        if thread_id:
            result = await send_fn(chat_id, text, thread_id=thread_id)
        else:
            result = await send_fn(chat_id, text)
        return f"[ok] Message sent via {platform}: {result[:200]}"
    except Exception as e:
        return f"[error sending via {platform}: {e}]"
