"""Reconnect backoff and TTL refresh.

Time is injected so these assert schedule, not sleep.
"""

from __future__ import annotations

import pytest

from baal_agent.mcp_client import MCPClient


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.mark.asyncio
async def test_failed_connect_backs_off_then_retries():
    clock = Clock()
    client = MCPClient(now=clock)
    attempts = []

    async def fake_connect(name, config):
        attempts.append(clock())
        raise RuntimeError("down")

    client.connect = fake_connect
    client._configured["srv"] = {"transport": "http", "url": "https://x/mcp"}

    await client.maintain()
    assert len(attempts) == 1
    await client.maintain()
    assert len(attempts) == 1, "retried before the backoff elapsed"
    clock.advance(31)
    await client.maintain()
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_backoff_doubles_and_caps():
    client = MCPClient(now=Clock())
    assert client._next_backoff(0) == 30
    assert client._next_backoff(30) == 60
    assert client._next_backoff(240) == 300
    assert client._next_backoff(300) == 300


@pytest.mark.asyncio
async def test_refresh_keeps_tools_when_listing_fails():
    clock = Clock()
    client = MCPClient(now=clock)
    calls = []

    async def failing_list(conn):
        calls.append(1)
        raise RuntimeError("blip")

    client._list_tools_for = failing_list
    conn = type("C", (), {"name": "srv", "tools": {"mcp_srv_a": {}}, "tools_listed_at": 0.0})()
    conn.transport = type("T", (), {"connected": True})()
    client.registry.register(conn, {"mcp_srv_a": {}})

    clock.advance(400)
    await client.maintain()
    assert calls, "TTL elapsed but no refresh attempted"
    assert client.registry.get("mcp_srv_a") is not None, "a failed refresh emptied the tools"
    assert conn.tools_listed_at == 0.0, "timestamp advanced despite failure"


@pytest.mark.asyncio
async def test_overlapping_maintain_does_not_double_run():
    import asyncio

    client = MCPClient(now=Clock())
    running = []

    async def slow_connect(name, config):
        running.append(1)
        await asyncio.sleep(0.05)

    client.connect = slow_connect
    client._configured["srv"] = {"transport": "http", "url": "https://x/mcp"}
    await asyncio.gather(client.maintain(), client.maintain())
    assert len(running) == 1
