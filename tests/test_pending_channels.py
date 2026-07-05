"""Per-channel pending cursors: two channels each see a message once,
acks are monotonic, the legacy destructive path is unchanged, and GC
removes only old rows."""

from __future__ import annotations

import pytest
import pytest_asyncio

from baal_agent.database import AgentDatabase


@pytest_asyncio.fixture
async def db(tmp_path):
    d = AgentDatabase(db_path=str(tmp_path / "agent.db"))
    await d.initialize()
    yield d
    await d.close()


@pytest.mark.asyncio
async def test_two_channels_each_get_message_once(db):
    await db.add_pending("__owner__", "hello", source="subagent")

    web = await db.get_pending_for_channel("web")
    tg = await db.get_pending_for_channel("telegram")
    assert [m["content"] for m in web] == ["hello"]
    assert [m["content"] for m in tg] == ["hello"]

    await db.ack_pending("web", web[-1]["id"])
    assert await db.get_pending_for_channel("web") == []
    # telegram cursor untouched
    assert len(await db.get_pending_for_channel("telegram")) == 1


@pytest.mark.asyncio
async def test_ack_is_monotonic(db):
    await db.add_pending("__owner__", "a")
    rows = await db.get_pending_for_channel("web")
    last = await db.ack_pending("web", rows[-1]["id"])
    assert await db.ack_pending("web", 0) == last  # cannot move backwards


@pytest.mark.asyncio
async def test_legacy_destructive_path_unchanged(db):
    await db.add_pending("__owner__", "x")
    got = await db.get_and_clear_pending()
    assert [m["content"] for m in got] == ["x"]
    assert await db.get_and_clear_pending() == []


@pytest.mark.asyncio
async def test_count_is_cursor_aware_per_channel(db):
    await db.add_pending("__owner__", "hello")
    assert await db.count_pending() == 1
    assert await db.count_pending(channel="web") == 1

    rows = await db.get_pending_for_channel("web")
    await db.ack_pending("web", rows[-1]["id"])

    # web channel has read everything; the raw queue still holds the row
    assert await db.count_pending(channel="web") == 0
    assert await db.count_pending(channel="telegram") == 1
    assert await db.count_pending() == 1


@pytest.mark.asyncio
async def test_gc_deletes_old_rows_only(db):
    await db.add_pending("__owner__", "old")
    await db.db.execute(
        "UPDATE pending_messages SET created_at = datetime('now', '-8 days')"
    )
    await db.db.commit()
    await db.add_pending("__owner__", "fresh")
    await db.ack_pending("web", 0)  # triggers GC
    remaining = await db.get_pending_for_channel("telegram")
    assert [m["content"] for m in remaining] == ["fresh"]
