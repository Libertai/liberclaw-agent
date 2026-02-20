"""Lightweight Telegram bot using raw httpx (no python-telegram-bot dependency)."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

import httpx

from baal_agent.database import AgentDatabase

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}"
POLL_TIMEOUT = 30  # long-polling timeout in seconds
MAX_MESSAGE_LENGTH = 4096


class TelegramBot:
    """Minimal Telegram bot that forwards messages to the agent."""

    def __init__(
        self,
        token: str,
        owner_telegram_id: str,
        db: AgentDatabase,
        agent_turn_callback: Callable[[str, str], Awaitable[str | None]],
    ) -> None:
        self.token = token
        self.owner_telegram_id = owner_telegram_id
        self.db = db
        self._agent_turn = agent_turn_callback
        self._base_url = API_BASE.format(token=token)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(POLL_TIMEOUT + 10))
        self._bot_id: int | None = None
        self._bot_username: str = ""
        self._bot_name: str = ""
        self._running = False

    # ── Telegram API helpers ──────────────────────────────────────────

    async def _api(self, method: str, **kwargs) -> dict:
        resp = await self._client.post(f"{self._base_url}/{method}", json=kwargs)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {data}")
        return data["result"]

    async def _send_message(self, chat_id: str | int, text: str, **kwargs) -> dict:
        # Split long messages
        chunks = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
        result = {}
        for chunk in chunks:
            result = await self._api("sendMessage", chat_id=chat_id, text=chunk, **kwargs)
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
                await self._send_typing(chat_id)
                tg_chat_id = f"tg:{chat_id}"
                try:
                    response = await self._agent_turn(text, tg_chat_id)
                    if response:
                        await self._send_message(chat_id, response)
                    else:
                        await self._send_message(chat_id, "(No response generated)")
                except Exception as e:
                    logger.error(f"Agent turn error for tg:{chat_id}: {e}")
                    await self._send_message(chat_id, "Sorry, an error occurred processing your message.")

        except Exception as e:
            logger.error(f"Telegram update handling error: {e}", exc_info=True)
