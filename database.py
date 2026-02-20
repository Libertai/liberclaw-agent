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

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str | None,
        *,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, tool_calls, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, tc_json, tool_call_id, now),
        )
        await self.db.commit()

    async def get_history(self, chat_id: str, limit: int = 50) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT role, content, tool_calls, tool_call_id "
            "FROM messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        messages = []
        for r in reversed(rows):
            msg: dict = {"role": r["role"]}
            if r["content"] is not None:
                msg["content"] = r["content"]
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            messages.append(msg)
        return messages

    async def compact_history(
        self, chat_id: str, keep_recent: int, summary: str
    ) -> None:
        """Replace old messages with a summary pair, keeping recent messages.

        1. Find the cutoff timestamp (the keep_recent-th most recent message)
        2. Delete all messages older than that cutoff
        3. Insert a user+assistant summary pair just before the cutoff
        """
        # Count total
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?", (chat_id,)
        )
        row = await cursor.fetchone()
        total = row["cnt"] if row else 0
        if total <= keep_recent:
            return

        # Find the cutoff: the created_at of the (keep_recent)-th most recent message
        cursor = await self.db.execute(
            "SELECT created_at FROM messages WHERE chat_id = ? "
            "ORDER BY created_at DESC LIMIT 1 OFFSET ?",
            (chat_id, keep_recent - 1),
        )
        cutoff_row = await cursor.fetchone()
        if not cutoff_row:
            return
        cutoff = cutoff_row["created_at"]

        # Delete old messages (strictly before the cutoff)
        await self.db.execute(
            "DELETE FROM messages WHERE chat_id = ? AND created_at < ?",
            (chat_id, cutoff),
        )

        # Find the earliest remaining message's timestamp to place summary before it
        cursor = await self.db.execute(
            "SELECT MIN(created_at) as earliest FROM messages WHERE chat_id = ?",
            (chat_id,),
        )
        earliest_row = await cursor.fetchone()
        earliest = earliest_row["earliest"] if earliest_row else cutoff

        # Insert summary pair just before the earliest remaining message.
        # Use timestamps that sort before the kept messages.
        # Parse earliest and subtract 2s / 1s to ensure ordering.
        from datetime import datetime, timedelta, timezone

        try:
            earliest_dt = datetime.fromisoformat(earliest)
        except (ValueError, TypeError):
            earliest_dt = datetime.now(timezone.utc)

        summary_user_ts = (earliest_dt - timedelta(seconds=2)).isoformat()
        summary_asst_ts = (earliest_dt - timedelta(seconds=1)).isoformat()

        await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, "user", f"[Earlier conversation summary]\n\n{summary}", summary_user_ts),
        )
        await self.db.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (
                chat_id,
                "assistant",
                "Understood, I have the context from our previous conversation.",
                summary_asst_ts,
            ),
        )
        await self.db.commit()

    async def clear_history(self, chat_id: str) -> int:
        """Delete all messages for a chat_id. Returns count deleted."""
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?", (chat_id,)
        )
        row = await cursor.fetchone()
        count = row["cnt"] if row else 0
        await self.db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await self.db.commit()
        return count

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
