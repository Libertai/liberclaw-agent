"""Lightweight Telegram bot using raw httpx (no python-telegram-bot dependency)."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable, Awaitable

import httpx

from baal_agent.database import AgentDatabase

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}"

# ── Telegram channel context ──────────────────────────────────────────

TELEGRAM_CHANNEL_HINT = (
    "\n\n## Output Channel\n\n"
    "You are responding on Telegram. Format your responses accordingly:\n"
    "- Use **bold** and `code` for emphasis, not tables or complex markdown\n"
    "- Tables do NOT render on Telegram — use bullet lists instead\n"
    "- Keep lines short — Telegram is mostly mobile\n"
    "- Use ``` for code blocks (they render as monospace)\n"
    "- Horizontal rules (---) don't render — use blank lines or emoji dividers\n"
    "- Headers (#) don't render — use **bold text** instead\n"
    "- Images/links work: [text](url)\n"
)


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_to_telegram_html(text: str) -> str:
    """Convert common markdown to Telegram-compatible HTML.

    Handles: bold, italic, code blocks, inline code, links, strikethrough.
    Falls back gracefully — if conversion produces bad HTML, the caller
    retries as plain text.
    """
    # Preserve code blocks first (don't process markdown inside them)
    code_blocks: list[str] = []

    def _stash_code_block(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = _html_escape(m.group(2))
        code_blocks.append(f"<pre>{code}</pre>")
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    result = re.sub(r"```(\w*)\n?(.*?)```", _stash_code_block, text, flags=re.DOTALL)

    # Inline code
    def _inline_code(m: re.Match) -> str:
        return f"<code>{_html_escape(m.group(1))}</code>"

    result = re.sub(r"`([^`\n]+)`", _inline_code, result)

    # Escape HTML in remaining text (but not our tags)
    # Split on our placeholders and HTML tags, escape only plain text parts
    parts = re.split(r"(\x00CODEBLOCK\d+\x00|</?(?:code|pre|b|i|s|a)[^>]*>)", result)
    for i, part in enumerate(parts):
        if not part.startswith(("\x00", "<")):
            parts[i] = _html_escape(part)
    result = "".join(parts)

    # Bold: **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"__(.+?)__", r"<b>\1</b>", result)

    # Italic: *text* or _text_ (but not inside words like file_name)
    result = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", result)
    result = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", result)

    # Strikethrough: ~~text~~
    result = re.sub(r"~~(.+?)~~", r"<s>\1</s>", result)

    # Links: [text](url)
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', result)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        result = result.replace(f"\x00CODEBLOCK{i}\x00", block)

    return result
POLL_TIMEOUT = 30  # long-polling timeout in seconds
MAX_MESSAGE_LENGTH = 4096
AGENT_TURN_TIMEOUT = 120  # max seconds for a single agent turn via Telegram
TYPING_REFRESH_INTERVAL = 4  # Telegram typing indicator expires after ~5s
MAX_QUEUED_MESSAGES = 3  # drop messages beyond this when chat is busy


class TelegramBot:
    """Minimal Telegram bot that forwards messages to the agent."""

    def __init__(
        self,
        token: str,
        owner_telegram_id: str,
        db: AgentDatabase,
        agent_turn_callback: Callable[[str, str], Awaitable[str | None]],
        cancel_run_callback: Callable[[str], bool] | None = None,
    ) -> None:
        self.token = token
        self.owner_telegram_id = owner_telegram_id
        self.db = db
        self._agent_turn = agent_turn_callback
        self._cancel_run = cancel_run_callback or (lambda chat_id: False)
        self._base_url = API_BASE.format(token=token)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(POLL_TIMEOUT + 10))
        self._bot_id: int | None = None
        self._bot_username: str = ""
        self._bot_name: str = ""
        self._running = False
        self._chat_locks: dict[str, asyncio.Lock] = {}  # per-chat serialization
        self._chat_queue_depth: dict[str, int] = {}  # track queued messages per chat

    # ── Telegram API helpers ──────────────────────────────────────────

    async def _api(self, method: str, **kwargs) -> dict:
        resp = await self._client.post(f"{self._base_url}/{method}", json=kwargs)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {data}")
        return data["result"]

    async def _send_message(self, chat_id: str | int, text: str, **kwargs) -> dict:
        """Send a message with HTML formatting, falling back to plain text on error."""
        html_text = _markdown_to_telegram_html(text)
        chunks = [html_text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(html_text), MAX_MESSAGE_LENGTH)]
        result = {}
        for chunk in chunks:
            try:
                result = await self._api(
                    "sendMessage", chat_id=chat_id, text=chunk,
                    parse_mode="HTML", **kwargs,
                )
            except Exception:
                # HTML parsing failed — strip tags and retry as plain text
                plain = re.sub(r"<[^>]+>", "", chunk)
                try:
                    result = await self._api(
                        "sendMessage", chat_id=chat_id, text=plain, **kwargs,
                    )
                except Exception as e:
                    logger.error(f"Failed to send message to {chat_id}: {e}")
        return result

    async def _send_typing(self, chat_id: str | int) -> None:
        try:
            await self._api("sendChatAction", chat_id=chat_id, action="typing")
        except Exception:
            pass  # non-critical

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Validate token, store bot info, auto-allow owner, delete webhook."""
        me = await self._api("getMe")
        self._bot_id = me["id"]
        self._bot_username = me.get("username", "")
        self._bot_name = me.get("first_name", "")
        logger.info(f"Telegram bot connected: @{self._bot_username} ({self._bot_name})")

        # Delete any existing webhook so getUpdates works
        await self._api("deleteWebhook")

        # Auto-allow owner
        if self.owner_telegram_id:
            existing = await self.db.get_telegram_contact(self.owner_telegram_id)
            if not existing:
                await self.db.upsert_telegram_contact(
                    telegram_id=self.owner_telegram_id,
                    chat_id=self.owner_telegram_id,
                    chat_type="private",
                    display_name="Owner",
                    username=None,
                    status="allowed",
                )
                logger.info(f"Auto-allowed owner telegram_id={self.owner_telegram_id}")

    async def poll_loop(self) -> None:
        """Long-polling loop with exponential backoff on errors."""
        self._running = True
        offset = 0
        backoff = 1

        while self._running:
            try:
                updates = await self._api(
                    "getUpdates",
                    offset=offset,
                    timeout=POLL_TIMEOUT,
                    allowed_updates=["message"],
                )
                backoff = 1  # reset on success

                for update in updates:
                    offset = update["update_id"] + 1
                    asyncio.create_task(self._handle_update(update))

            except asyncio.CancelledError:
                self._running = False
                return
            except Exception as e:
                logger.error(f"Telegram poll error: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def stop(self) -> None:
        self._running = False
        await self._client.aclose()

    @property
    def bot_username(self) -> str:
        return self._bot_username

    @property
    def bot_name(self) -> str:
        return self._bot_name

    @property
    def connected(self) -> bool:
        return self._bot_id is not None

    # ── Agent turn with typing indicator ─────────────────────────────

    async def _run_agent_with_typing(
        self, chat_id: str, tg_chat_id: str, text: str
    ) -> None:
        """Run an agent turn with a persistent typing indicator and timeout.

        - Refreshes the typing indicator every few seconds so the user
          sees activity during long tool loops / inference calls.
        - Enforces a wall-clock timeout to prevent infinite hangs.
        - Serialization (per-chat lock) is handled by the caller.
        """
        typing_task: asyncio.Task | None = None

        async def _keep_typing():
            """Refresh typing indicator until cancelled."""
            try:
                while True:
                    await self._send_typing(chat_id)
                    await asyncio.sleep(TYPING_REFRESH_INTERVAL)
            except asyncio.CancelledError:
                return

        try:
            # Start persistent typing indicator
            typing_task = asyncio.create_task(_keep_typing())

            # Run the agent turn with a timeout
            response = await asyncio.wait_for(
                self._agent_turn(text, tg_chat_id),
                timeout=AGENT_TURN_TIMEOUT,
            )

            if response:
                await self._send_message(chat_id, response)
            else:
                await self._send_message(chat_id, "(No response generated)")

        except asyncio.TimeoutError:
            logger.warning(
                f"Agent turn timed out for {tg_chat_id} after {AGENT_TURN_TIMEOUT}s"
            )
            await self._send_message(
                chat_id,
                "Sorry, the response took too long. Please try again.",
            )
        except Exception as e:
            logger.error(f"Agent turn error for {tg_chat_id}: {e}", exc_info=True)
            await self._send_message(
                chat_id,
                "Sorry, an error occurred processing your message.",
            )
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()

    # ── Update handling ───────────────────────────────────────────────

    async def _handle_update(self, update: dict) -> None:
        try:
            msg = update.get("message")
            if not msg:
                return

            text = msg.get("text", "")
            if not text:
                return

            chat = msg["chat"]
            chat_id = str(chat["id"])
            chat_type = chat.get("type", "private")
            user = msg.get("from", {})
            user_id = str(user.get("id", ""))
            display_name = " ".join(
                filter(None, [user.get("first_name", ""), user.get("last_name", "")])
            ) or user_id
            username = user.get("username")

            # In groups: only respond to @mentions or replies to the bot
            if chat_type in ("group", "supergroup"):
                is_mention = self._bot_username and f"@{self._bot_username}" in text
                is_reply_to_bot = (
                    msg.get("reply_to_message", {}).get("from", {}).get("id") == self._bot_id
                )
                if not is_mention and not is_reply_to_bot:
                    return
                # Strip @botname from text
                if self._bot_username:
                    text = text.replace(f"@{self._bot_username}", "").strip()

            if not text:
                return

            # Contact ID: user_id for DMs, chat_id for groups
            contact_id = user_id if chat_type == "private" else chat_id

            contact = await self.db.get_telegram_contact(contact_id)

            if contact is None:
                # New contact — register as pending
                await self.db.upsert_telegram_contact(
                    telegram_id=contact_id,
                    chat_id=chat_id,
                    chat_type=chat_type,
                    display_name=display_name,
                    username=username,
                    status="pending",
                )
                # Notify the owner via pending messages
                await self.db.add_pending(
                    "__owner__",
                    f"New Telegram contact requesting access: "
                    f"{display_name} (@{username or 'N/A'}, id: {contact_id}, type: {chat_type})",
                    source="telegram",
                )
                await self._send_message(
                    chat_id,
                    "Your message has been received. Access is pending approval by the agent owner.",
                )
                return

            status = contact["status"]

            if status == "blocked":
                return

            if status == "pending":
                await self._send_message(
                    chat_id,
                    "Your access is still pending approval. Please wait.",
                )
                return

            if status == "allowed":
                tg_chat_id = f"tg:{chat_id}"

                # Handle /stop command — cancel active run and clear queue
                if text.strip().lower() in ("/stop", "/stop@" + self._bot_username.lower()):
                    cancelled = False
                    lock = self._chat_locks.get(tg_chat_id)
                    if lock and lock.locked():
                        # Cancel queued messages by maxing out depth
                        self._chat_queue_depth[tg_chat_id] = MAX_QUEUED_MESSAGES + 1
                        cancelled = True
                    # Cancel the active agent turn
                    if self._cancel_run(tg_chat_id):
                        cancelled = True
                    if cancelled:
                        await self._send_message(chat_id, "⏹ Stopped. What's next?")
                    else:
                        await self._send_message(chat_id, "Nothing running right now.")
                    # Reset queue depth so new messages flow through
                    self._chat_queue_depth.pop(tg_chat_id, None)
                    return

                # Serialize messages per chat to prevent concurrent turns
                # from corrupting history or producing duplicate responses
                lock = self._chat_locks.setdefault(tg_chat_id, asyncio.Lock())

                if lock.locked():
                    # Backpressure: drop if too many messages queued
                    depth = self._chat_queue_depth.get(tg_chat_id, 0)
                    if depth >= MAX_QUEUED_MESSAGES:
                        logger.warning(f"Chat {tg_chat_id} queue full ({depth}), dropping message")
                        return
                    logger.info(f"Chat {tg_chat_id} is busy, queuing message ({depth + 1})")

                self._chat_queue_depth[tg_chat_id] = self._chat_queue_depth.get(tg_chat_id, 0) + 1
                try:
                    async with lock:
                        await self._run_agent_with_typing(chat_id, tg_chat_id, text)
                finally:
                    self._chat_queue_depth[tg_chat_id] = max(0, self._chat_queue_depth.get(tg_chat_id, 1) - 1)
                    # Clean up locks for idle chats
                    if not lock.locked() and self._chat_queue_depth.get(tg_chat_id, 0) == 0:
                        self._chat_locks.pop(tg_chat_id, None)
                        self._chat_queue_depth.pop(tg_chat_id, None)

        except Exception as e:
            logger.error(f"Telegram update handling error: {e}", exc_info=True)
