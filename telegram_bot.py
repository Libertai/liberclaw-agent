"""Lightweight Telegram bot using raw httpx (no python-telegram-bot dependency)."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Callable, Awaitable

import httpx

from baal_agent.database import AgentDatabase
from baal_agent.image_utils import is_image, build_image_content_blocks

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
        agent_turn_callback: Callable[[str | list[dict], str], Awaitable[tuple[str | None, list[dict]]]],
        cancel_run_callback: Callable[[str], bool] | None = None,
        workspace_path: str = "/tmp/workspace",
    ) -> None:
        self.token = token
        self.owner_telegram_id = owner_telegram_id
        self.db = db
        self._agent_turn = agent_turn_callback
        self._cancel_run = cancel_run_callback or (lambda chat_id: False)
        self._workspace_path = workspace_path
        self._base_url = API_BASE.format(token=token)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(POLL_TIMEOUT + 10))
        self._bot_id: int | None = None
        self._bot_username: str = ""
        self._bot_name: str = ""
        self._running = False
        self._chat_locks: dict[str, asyncio.Lock] = {}  # per-chat serialization
        self._chat_queue_depth: dict[str, int] = {}  # track queued messages per chat
        self._media_groups: dict[str, list[dict]] = {}  # media_group_id -> list of messages
        self._media_group_tasks: dict[str, asyncio.Task] = {}  # media_group_id -> flush task

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

    async def _send_local_file(
        self, chat_id: str | int, rel_path: str, caption: str = ""
    ) -> None:
        """Send a file from the agent workspace to Telegram."""
        workspace = Path(self._workspace_path)
        file_path = (workspace / rel_path).resolve()
        # Security: ensure path is within workspace
        try:
            file_path.relative_to(workspace.resolve())
        except ValueError:
            logger.warning(f"Refusing to send file outside workspace: {rel_path}")
            return
        if not file_path.is_file():
            logger.warning(f"File not found for Telegram send: {file_path}")
            return

        filename = file_path.name
        file_bytes = file_path.read_bytes()
        tg_caption = _markdown_to_telegram_html(caption) if caption else None

        try:
            if is_image(str(file_path)) and len(file_bytes) <= 10 * 1024 * 1024:
                data = {"chat_id": str(chat_id), "parse_mode": "HTML"}
                if tg_caption:
                    data["caption"] = tg_caption
                resp = await self._client.post(
                    f"{self._base_url}/sendPhoto",
                    data=data,
                    files={"photo": (filename, file_bytes)},
                )
            else:
                data = {"chat_id": str(chat_id), "parse_mode": "HTML"}
                if tg_caption:
                    data["caption"] = tg_caption
                resp = await self._client.post(
                    f"{self._base_url}/sendDocument",
                    data=data,
                    files={"document": (filename, file_bytes)},
                )
            result = resp.json()
            if not result.get("ok"):
                # Retry caption without HTML if it failed
                if tg_caption:
                    plain_caption = re.sub(r"<[^>]+>", "", tg_caption)
                    data["caption"] = plain_caption
                    del data["parse_mode"]
                    if is_image(str(file_path)) and len(file_bytes) <= 10 * 1024 * 1024:
                        resp = await self._client.post(
                            f"{self._base_url}/sendPhoto",
                            data=data,
                            files={"photo": (filename, file_bytes)},
                        )
                    else:
                        resp = await self._client.post(
                            f"{self._base_url}/sendDocument",
                            data=data,
                            files={"document": (filename, file_bytes)},
                        )
                else:
                    logger.error(f"Telegram sendPhoto/Document failed: {result}")
        except Exception as e:
            logger.error(f"Failed to send file {rel_path} to {chat_id}: {e}")

    async def _send_typing(self, chat_id: str | int) -> None:
        try:
            await self._api("sendChatAction", chat_id=chat_id, action="typing")
        except Exception:
            pass  # non-critical

    async def _download_file(self, file_id: str, filename: str) -> str:
        """Download a file from Telegram and save it to the workspace uploads dir."""
        file_info = await self._api("getFile", file_id=file_id)
        file_path = file_info["file_path"]
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        resp = await self._client.get(url)
        resp.raise_for_status()

        uploads_dir = Path(self._workspace_path) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        local_path = uploads_dir / filename
        local_path.write_bytes(resp.content)
        return str(local_path)

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
        self, chat_id: str, tg_chat_id: str, text: str | list[dict]
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
            response, file_events = await asyncio.wait_for(
                self._agent_turn(text, tg_chat_id),
                timeout=AGENT_TURN_TIMEOUT,
            )

            # Send any generated files (images, documents)
            for fev in file_events:
                await self._send_local_file(
                    chat_id, fev.get("path", ""), fev.get("caption", ""),
                )

            if response:
                await self._send_message(chat_id, response)
            elif not file_events:
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

    # ── Media group handling ─────────────────────────────────────────

    async def _flush_media_group(
        self, media_group_id: str, chat_id: str, tg_chat_id: str
    ) -> None:
        """Wait for all media in a group to arrive, then process as one turn."""
        await asyncio.sleep(0.5)  # Telegram sends media group messages in quick succession

        messages = self._media_groups.pop(media_group_id, [])
        self._media_group_tasks.pop(media_group_id, None)
        if not messages:
            return

        content_blocks: list[dict] = []
        captions: list[str] = []

        for msg in messages:
            caption = msg.get("caption", "")
            if caption:
                captions.append(caption)

            # Try photo first, then document
            if msg.get("photo"):
                photo = msg["photo"][-1]  # highest resolution
                file_id = photo["file_id"]
                try:
                    local_path = await self._download_file(file_id, f"{file_id}.jpg")
                    content_blocks.extend(build_image_content_blocks(local_path))
                except Exception as e:
                    logger.error(f"Failed to download photo from media group: {e}")
            elif msg.get("document"):
                doc = msg["document"]
                file_id = doc["file_id"]
                file_name = doc.get("file_name", file_id)
                try:
                    local_path = await self._download_file(file_id, file_name)
                    if is_image(local_path):
                        content_blocks.extend(build_image_content_blocks(local_path))
                    else:
                        content_blocks.append({
                            "type": "text",
                            "text": f"[File saved: {local_path}]",
                        })
                except Exception as e:
                    logger.error(f"Failed to download document from media group: {e}")

        combined_caption = "\n".join(captions).strip()
        if combined_caption:
            content_blocks.insert(0, {"type": "text", "text": combined_caption})
        elif not any(b.get("type") == "text" for b in content_blocks):
            content_blocks.insert(0, {"type": "text", "text": "[User sent images]"})

        if not content_blocks:
            return

        # Use the per-chat lock for serialization
        lock = self._chat_locks.setdefault(tg_chat_id, asyncio.Lock())
        self._chat_queue_depth[tg_chat_id] = self._chat_queue_depth.get(tg_chat_id, 0) + 1
        try:
            async with lock:
                await self._run_agent_with_typing(chat_id, tg_chat_id, content_blocks)
        finally:
            self._chat_queue_depth[tg_chat_id] = max(0, self._chat_queue_depth.get(tg_chat_id, 1) - 1)
            if not lock.locked() and self._chat_queue_depth.get(tg_chat_id, 0) == 0:
                self._chat_locks.pop(tg_chat_id, None)
                self._chat_queue_depth.pop(tg_chat_id, None)

    # ── Update handling ───────────────────────────────────────────────

    async def _handle_update(self, update: dict) -> None:
        try:
            msg = update.get("message")
            if not msg:
                return

            text = msg.get("text", "")
            has_photo = bool(msg.get("photo"))
            has_document = bool(msg.get("document"))
            has_media = has_photo or has_document

            if not text and not has_media:
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
                caption = msg.get("caption", "")
                searchable = text or caption
                is_mention = self._bot_username and f"@{self._bot_username}" in searchable
                is_reply_to_bot = (
                    msg.get("reply_to_message", {}).get("from", {}).get("id") == self._bot_id
                )
                if not is_mention and not is_reply_to_bot:
                    return
                # Strip @botname from text/caption
                if self._bot_username:
                    text = text.replace(f"@{self._bot_username}", "").strip()

            if not text and not has_media:
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

                # Handle /clear command — wipe conversation history
                if text and text.strip().lower() in ("/clear", "/clear@" + self._bot_username.lower()):
                    count = await self.db.clear_history(tg_chat_id)
                    await self._send_message(
                        chat_id,
                        f"Conversation cleared ({count} messages removed). Starting fresh.",
                    )
                    return

                # Handle /stop command — cancel active run and clear queue
                if text and text.strip().lower() in ("/stop", "/stop@" + self._bot_username.lower()):
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

                # ── Media group handling ──
                media_group_id = msg.get("media_group_id")
                if media_group_id and has_media:
                    self._media_groups.setdefault(media_group_id, []).append(msg)
                    # Cancel any existing flush task and reschedule
                    existing_task = self._media_group_tasks.get(media_group_id)
                    if existing_task and not existing_task.done():
                        existing_task.cancel()
                    self._media_group_tasks[media_group_id] = asyncio.create_task(
                        self._flush_media_group(media_group_id, chat_id, tg_chat_id)
                    )
                    return

                # ── Single photo handling ──
                content: str | list[dict] = text
                if has_photo and not media_group_id:
                    photo = msg["photo"][-1]  # highest resolution
                    caption = msg.get("caption", "") or text
                    try:
                        local_path = await self._download_file(photo["file_id"], f"{photo['file_id']}.jpg")
                        content = build_image_content_blocks(local_path, annotation=caption or "[User sent image]")
                    except Exception as e:
                        logger.error(f"Failed to download photo: {e}")
                        content = caption or "[Photo could not be downloaded]"

                # ── Single document handling ──
                elif has_document and not media_group_id:
                    doc = msg["document"]
                    file_name = doc.get("file_name", doc["file_id"])
                    caption = msg.get("caption", "") or text
                    try:
                        local_path = await self._download_file(doc["file_id"], file_name)
                        if is_image(local_path):
                            content = build_image_content_blocks(local_path, annotation=caption or f"[User sent image: {file_name}]")
                        else:
                            annotation = f"[File saved: {local_path}]"
                            if caption:
                                annotation = f"{caption}\n{annotation}"
                            content = annotation
                    except Exception as e:
                        logger.error(f"Failed to download document: {e}")
                        content = caption or "[Document could not be downloaded]"

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
                        await self._run_agent_with_typing(chat_id, tg_chat_id, content)
                finally:
                    self._chat_queue_depth[tg_chat_id] = max(0, self._chat_queue_depth.get(tg_chat_id, 1) - 1)
                    # Clean up locks for idle chats
                    if not lock.locked() and self._chat_queue_depth.get(tg_chat_id, 0) == 0:
                        self._chat_locks.pop(tg_chat_id, None)
                        self._chat_queue_depth.pop(tg_chat_id, None)

        except Exception as e:
            logger.error(f"Telegram update handling error: {e}", exc_info=True)
