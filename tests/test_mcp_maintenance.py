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
    """The lock alone still serialises an overlapping tick behind the one in
    flight -- it just makes the second tick wait, then run for real, seeing
    whatever state the first tick left behind (e.g. already connected,
    already backed off). That's not the guard; it's incidental and hides a
    removed guard. So the body maintain() calls is patched directly, with
    nothing else -- like backoff or a connected check -- that could mask a
    second call: only the `_maintaining` checks can suppress it.
    """
    import asyncio

    client = MCPClient(now=Clock())
    calls = []

    async def slow_reconnect():
        calls.append(1)
        await asyncio.sleep(0.05)

    async def noop_refresh():
        pass

    client._reconnect_unreachable = slow_reconnect
    client._refresh_stale_tools = noop_refresh

    await asyncio.gather(client.maintain(), client.maintain())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_dead_connected_server_gets_reconnected():
    """A server that connects, then drops (transport.connected flips False
    without the conn ever leaving the registry), must be reconnected — not
    skipped just because a (dead) conn is still in registry.servers.
    """
    clock = Clock()
    client = MCPClient(now=clock)
    closed = []

    class DeadTransport:
        connected = False

        async def close(self):
            closed.append("srv")

    conn = type("C", (), {"name": "srv", "tools": {}, "tools_listed_at": 0.0})()
    conn.transport = DeadTransport()
    client.registry.register(conn, {})
    client._configured["srv"] = {"transport": "http", "url": "https://x/mcp"}

    attempts = []

    async def fake_connect(name, config):
        attempts.append(name)

    client.connect = fake_connect

    await client.maintain()
    assert closed == ["srv"], "the dead transport was never torn down"
    assert attempts == ["srv"], "reconnect wasn't attempted for a registered-but-dead server"


@pytest.mark.asyncio
async def test_dead_http_transport_gets_reconnected():
    """Same as above but the "dead" signal comes from a real HttpTransport
    whose connected flips False after an actual connection-level failure,
    not a stub — proves the transport's own health flag, not just the
    maintenance predicate.
    """
    import httpx

    from baal_agent.mcp_http import HttpTransport
    from baal_agent.mcp_transport import MCPError as _MCPError

    def failing_handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    transport = HttpTransport("https://x/mcp")
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    with pytest.raises(_MCPError):
        await transport.send_request("tools/list", {}, 5.0)
    assert not transport.connected, "connected must go False after a transport-level failure"

    clock = Clock()
    client = MCPClient(now=clock)
    conn = type("C", (), {"name": "srv", "tools": {}, "tools_listed_at": 0.0})()
    conn.transport = transport
    client.registry.register(conn, {})
    client._configured["srv"] = {"transport": "http", "url": "https://x/mcp"}

    attempts = []

    async def fake_connect(name, config):
        attempts.append(name)

    client.connect = fake_connect

    await client.maintain()
    assert attempts == ["srv"], "a dead HttpTransport must trigger a reconnect"
    await transport.close()


@pytest.mark.asyncio
async def test_reinit_resets_tools_listed_at():
    """A session-recovery re-init must reset the TTL clock, or the next
    maintenance tick re-lists the same server again shortly after.
    """
    clock = Clock()
    client = MCPClient(now=clock)

    class FakeTransport:
        async def send_request(self, method, params, timeout=30.0):
            return {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}

        async def send_notification(self, method, params):
            pass

    conn = type("C", (), {"name": "srv", "tools": {}, "tools_listed_at": 0.0})()
    client.registry.register(conn, {})

    async def fake_list_all_tools(name, transport):
        return {"mcp_srv_a": {}}

    client._list_all_tools = fake_list_all_tools

    clock.advance(500)
    await client._reinit_server("srv", FakeTransport(), conn)
    assert conn.tools_listed_at == clock()


@pytest.mark.asyncio
async def test_shutdown_cancels_maintenance_before_disconnect(monkeypatch):
    """Mirrors the reviewer's probe: a tick blocked mid-reconnect must be
    cancelled and awaited before disconnect_all() runs, so it can never
    resume past its await and re-register a connection after teardown.
    """
    import asyncio

    from baal_agent import tools

    monkeypatch.setattr(tools, "_MCP_MAINTENANCE_INTERVAL", 0.0)
    monkeypatch.setattr(tools, "_mcp_maintenance_task", None)

    client = MCPClient(now=Clock())

    # An already-connected server: proves disconnect_all still tears down
    # real connections once the maintenance task is out of the way.
    closed = []

    class LiveTransport:
        connected = True

        async def close(self):
            closed.append("live")

    live_conn = type("C", (), {"name": "live", "tools": {}, "tools_listed_at": 0.0})()
    live_conn.transport = LiveTransport()
    client.registry.register(live_conn, {})
    client._configured["live"] = {"transport": "http", "url": "https://live/mcp"}

    # A dead server whose reconnect is in flight when shutdown starts.
    started = asyncio.Event()
    never = asyncio.Event()

    async def slow_connect(name, config):
        started.set()
        await never.wait()
        # Only reachable if the tick outlives cancellation — must not happen.
        conn = type("C", (), {"name": name, "tools": {}, "tools_listed_at": 0.0})()
        conn.transport = type("T", (), {"connected": True})()
        client.registry.register(conn, {})

    client.connect = slow_connect
    client._configured["srv"] = {"transport": "http", "url": "https://x/mcp"}

    monkeypatch.setattr(tools, "_mcp_client", client)
    await tools.start_mcp_maintenance()
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await tools.shutdown_mcp()

    assert tools._mcp_maintenance_task is None, "maintenance task still pending after shutdown"
    assert "srv" not in client.registry.servers, (
        "a tick that outlived cancellation re-registered a server after teardown"
    )
    assert client.registry.servers == {}
    assert closed == ["live"], "disconnect_all never tore down the live server"
