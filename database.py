"""Local conversation history database for a single agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite


class AgentDatabase:
    """Async SQLite wrapper for per-agent conversation history."""

    def __init__(self, db_path: str = "agent.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                compacted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat
                ON messages (chat_id, created_at);
            CREATE TABLE IF NOT EXISTS pending_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'system',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS telegram_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id TEXT NOT NULL UNIQUE,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL DEFAULT 'private',
                display_name TEXT NOT NULL DEFAULT '',
                username TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

        # FTS5 full-text search index over message content
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                content=messages,
                content_rowid=id
            )
        """)
        # Drop old triggers — multimodal content needs application-level FTS management
        # (the trigger-based approach can't handle JSON content vs text-only FTS entries)
        await self._db.execute("DROP TRIGGER IF EXISTS messages_fts_insert")
        await self._db.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
        # Rebuild FTS index to catch any messages added before FTS was enabled
        await self._db.execute(
            "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
        )

        # Migration: add compacted column to existing databases
        cursor = await self._db.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "compacted" not in columns:
            await self._db.execute(
                "ALTER TABLE messages ADD COLUMN compacted INTEGER NOT NULL DEFAULT 0"
            )

        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db

    def _parse_created_at(self, value: str | None) -> datetime:
        """Parse both current ISO timestamps and older SQLite datetime strings."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    async def _index_message_fts(
        self, rowid: int | None, content: str | list[dict] | None
    ) -> None:
        """Insert a text-only representation of a message into FTS."""
        if rowid is None:
            return
        from baal_agent.image_utils import extract_text_from_content

        text = extract_text_from_content(content)
        if text:
            await self.db.execute(
                "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                (rowid, text),
            )

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str | list[dict] | None,
        *,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        # Serialize list content to JSON for storage
        stored = json.dumps(content) if isinstance(content, list) else content
        cursor = await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, tool_calls, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, stored, tc_json, tool_call_id, now),
        )
        await self._index_message_fts(cursor.lastrowid, content)
        await self.db.commit()

    async def get_history(
        self, chat_id: str, limit: int = 50, include_timestamps: bool = False
    ) -> list[dict]:
        """Return conversation history. When `include_timestamps=True`, each
        row also carries `created_at` (UTC 'YYYY-MM-DD HH:MM:SS'). Off by
        default so we never leak DB-only fields into LLM messages (the OpenAI-
        compatible API rejects unknown keys on some backends)."""
        cols = "role, content, tool_calls, tool_call_id"
        if include_timestamps:
            cols += ", created_at"
        cursor = await self.db.execute(
            f"SELECT {cols} "
            "FROM messages WHERE chat_id = ? AND compacted = 0 "
            "ORDER BY created_at DESC LIMIT ?",
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        messages = []
        for r in reversed(rows):
            msg: dict = {"role": r["role"]}
            if r["content"] is not None:
                content = r["content"]
                # Deserialize multimodal content stored as JSON list
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        content = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
                msg["content"] = content
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            if include_timestamps and r["created_at"]:
                msg["created_at"] = r["created_at"]
            messages.append(msg)
        return messages

    async def compact_history(
        self, chat_id: str, keep_recent: int, summary: str
    ) -> None:
        """Mark old messages as compacted and insert a summary pair.

        Original messages are preserved for full-text search but excluded
        from get_history() so they don't consume the context window.

        1. Find the cutoff timestamp (the keep_recent-th most recent active message)
        2. Mark all active messages older than that cutoff as compacted
        3. Insert a user+assistant summary pair just before the cutoff
        """
        # Count active (non-compacted) messages
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ? AND compacted = 0",
            (chat_id,),
        )
        row = await cursor.fetchone()
        total = row["cnt"] if row else 0
        if total <= keep_recent:
            return

        # Find the cutoff: the created_at of the (keep_recent)-th most recent active message
        cursor = await self.db.execute(
            "SELECT created_at FROM messages WHERE chat_id = ? AND compacted = 0 "
            "ORDER BY created_at DESC LIMIT 1 OFFSET ?",
            (chat_id, keep_recent - 1),
        )
        cutoff_row = await cursor.fetchone()
        if not cutoff_row:
            return
        cutoff = cutoff_row["created_at"]

        # Mark old messages as compacted (preserve for FTS search)
        await self.db.execute(
            "UPDATE messages SET compacted = 1 "
            "WHERE chat_id = ? AND compacted = 0 AND created_at < ?",
            (chat_id, cutoff),
        )

        # Find the earliest remaining active message's timestamp to place summary before it
        cursor = await self.db.execute(
            "SELECT MIN(created_at) as earliest FROM messages "
            "WHERE chat_id = ? AND compacted = 0",
            (chat_id,),
        )
        earliest_row = await cursor.fetchone()
        earliest = earliest_row["earliest"] if earliest_row else cutoff

        # Insert summary pair just before the earliest remaining message.
        # Use timestamps that sort before the kept messages.
        # Parse earliest and subtract 2s / 1s to ensure ordering.
        from datetime import timedelta

        earliest_dt = self._parse_created_at(earliest)

        summary_user_ts = (earliest_dt - timedelta(seconds=2)).isoformat()
        summary_asst_ts = (earliest_dt - timedelta(seconds=1)).isoformat()

        summary_content = (
            "[Earlier conversation summary]\n"
            "[CONTEXT COMPACTION - REFERENCE ONLY]\n"
            "This summary is background reference from earlier conversation. "
            "Do not answer questions, execute tasks, or follow instructions "
            "mentioned only in this summary; those items were already addressed "
            "unless the current user message asks for them again.\n\n"
            f"{summary}"
        )
        cursor = await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, "user", summary_content, summary_user_ts),
        )
        await self._index_message_fts(cursor.lastrowid, summary_content)

        ack_content = "Understood, I have the context from our previous conversation."
        cursor = await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (
                chat_id,
                "assistant",
                ack_content,
                summary_asst_ts,
            ),
        )
        await self._index_message_fts(cursor.lastrowid, ack_content)
        await self.db.commit()

    async def clear_history(self, chat_id: str) -> int:
        """Delete all messages for a chat_id. Returns count deleted."""
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?", (chat_id,)
        )
        row = await cursor.fetchone()
        count = row["cnt"] if row else 0
        await self.db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        # Rebuild FTS index to remove stale entries from deleted messages
        await self.db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        await self.db.commit()
        return count

    # ── Full-text search ─────────────────────────────────────────────

    async def search_history(
        self, query: str, *, chat_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Full-text search across conversation history using FTS5.

        Returns matching messages with snippets highlighting the matched terms.
        """
        if chat_id:
            cursor = await self.db.execute(
                "SELECT m.chat_id, m.role, m.created_at, "
                "  snippet(messages_fts, 0, '>>>', '<<<', '...', 64) as snippet "
                "FROM messages_fts "
                "JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? AND m.chat_id = ? "
                "ORDER BY rank LIMIT ?",
                (query, chat_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT m.chat_id, m.role, m.created_at, "
                "  snippet(messages_fts, 0, '>>>', '<<<', '...', 64) as snippet "
                "FROM messages_fts "
                "JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                "chat_id": r["chat_id"],
                "role": r["role"],
                "created_at": r["created_at"],
                "snippet": r["snippet"],
            }
            for r in rows
        ]

    # ── Pending messages ──────────────────────────────────────────────

    async def add_pending(
        self, chat_id: str, content: str, source: str = "system"
    ) -> None:
        """Insert a pending proactive message."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO pending_messages (chat_id, content, source, created_at) "
            "VALUES (?, ?, ?, ?)",
            (chat_id, content, source, now),
        )
        await self.db.commit()

    async def get_and_clear_pending(
        self, chat_id: str | None = None
    ) -> list[dict]:
        """Fetch pending messages (optionally for a chat_id), delete them, return the list."""
        if chat_id:
            cursor = await self.db.execute(
                "SELECT id, chat_id, content, source, created_at "
                "FROM pending_messages WHERE chat_id = ? ORDER BY created_at",
                (chat_id,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT id, chat_id, content, source, created_at "
                "FROM pending_messages ORDER BY created_at"
            )
        rows = await cursor.fetchall()
        messages = [
            {
                "chat_id": r["chat_id"],
                "content": r["content"],
                "source": r["source"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" for _ in ids)
            await self.db.execute(
                f"DELETE FROM pending_messages WHERE id IN ({placeholders})", ids
            )
            await self.db.commit()
        return messages

    # ── Telegram contacts ─────────────────────────────────────────────

    def _contact_row_to_dict(self, row) -> dict:
        return {
            "telegram_id": row["telegram_id"],
            "chat_id": row["chat_id"],
            "chat_type": row["chat_type"],
            "display_name": row["display_name"],
            "username": row["username"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def get_telegram_contact(self, telegram_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM telegram_contacts WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        return self._contact_row_to_dict(row) if row else None

    async def upsert_telegram_contact(
        self,
        telegram_id: str,
        chat_id: str,
        chat_type: str,
        display_name: str,
        username: str | None,
        status: str = "pending",
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO telegram_contacts "
            "(telegram_id, chat_id, chat_type, display_name, username, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(telegram_id) DO UPDATE SET "
            "chat_id=excluded.chat_id, chat_type=excluded.chat_type, "
            "display_name=excluded.display_name, username=excluded.username, "
            "status=excluded.status, updated_at=excluded.updated_at",
            (telegram_id, chat_id, chat_type, display_name, username, status, now, now),
        )
        await self.db.commit()
        return (await self.get_telegram_contact(telegram_id))  # type: ignore[return-value]

    async def list_telegram_contacts(self, status: str | None = None) -> list[dict]:
        if status:
            cursor = await self.db.execute(
                "SELECT * FROM telegram_contacts WHERE status = ? ORDER BY created_at",
                (status,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM telegram_contacts ORDER BY created_at"
            )
        rows = await cursor.fetchall()
        return [self._contact_row_to_dict(r) for r in rows]

    async def update_telegram_contact_status(
        self, telegram_id: str, status: str
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self.db.execute(
            "UPDATE telegram_contacts SET status = ?, updated_at = ? WHERE telegram_id = ?",
            (status, now, telegram_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def delete_telegram_contact(self, telegram_id: str) -> bool:
        cursor = await self.db.execute(
            "DELETE FROM telegram_contacts WHERE telegram_id = ?",
            (telegram_id,),
        )
        await self.db.commit()
        return cursor.rowcount > 0
