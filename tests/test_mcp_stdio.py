"""Behaviour of the stdio MCP transport, pinned before it is extracted.

These assert what the client does today so the extraction can be verified as a
pure move. The fixture runs under `python -u`: a block-buffered pipe would stall
readline() until the request timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from baal_agent.mcp_client import MCPClient

FIXTURE = str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")


def stdio_config(**overrides):
    config = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-u", FIXTURE],
        "env": {"PATH": "/usr/bin:/bin"},
    }
    config.update(overrides)
    return config


@pytest.mark.xfail(strict=True, reason="tools/list pagination lands in Task 4")
@pytest.mark.asyncio
async def test_connect_registers_namespaced_tools():
    client = MCPClient()
    await client.connect("probe", stdio_config())
    try:
        names = {d["function"]["name"] for d in client.get_tool_definitions()}
        assert "mcp_probe_alpha" in names
        assert "mcp_probe_beta" in names, "second page of tools/list was dropped"
    finally:
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_call_tool_returns_server_text():
    client = MCPClient()
    await client.connect("probe", stdio_config())
    try:
        result = await client.call_tool_result("mcp_probe_alpha", {})
        assert result.is_error is False
        assert "called alpha" in result.content
    finally:
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_health_reports_connected_server():
    client = MCPClient()
    await client.connect("probe", stdio_config())
    try:
        health = client.get_health()
        server = next(s for s in health["servers"] if s["name"] == "probe")
        assert server["connected"] is True
        assert server["transport"] == "stdio"
    finally:
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_missing_command_is_recorded_not_raised():
    client = MCPClient()
    await client.connect("broken", stdio_config(command="/nonexistent/binary"))
    health = client.get_health()
    server = next(s for s in health["servers"] if s["name"] == "broken")
    assert server["connected"] is False
    assert server["error"]


@pytest.mark.asyncio
async def test_unknown_transport_is_recorded():
    client = MCPClient()
    await client.connect("weird", {"transport": "carrier-pigeon"})
    health = client.get_health()
    server = next(s for s in health["servers"] if s["name"] == "weird")
    assert server["connected"] is False
    assert "carrier-pigeon" in server["error"]


@pytest.mark.asyncio
async def test_disconnect_stops_the_subprocess():
    client = MCPClient()
    await client.connect("probe", stdio_config())
    conn = client._servers["probe"]
    process = conn.transport._process
    await client.disconnect_all()
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_tools_list_null_result_leaves_connection_open():
    """A malformed tools/list response is a soft failure: the server already
    initialized fine, so the connection survives with zero tools and no error
    recorded — same as an empty tools list."""
    client = MCPClient()
    await client.connect("probe", stdio_config(env={
        "PATH": "/usr/bin:/bin", "FAKE_MCP_TOOLS_LIST_MODE": "null",
    }))
    try:
        health = client.get_health()
        server = next(s for s in health["servers"] if s["name"] == "probe")
        assert server["connected"] is True
        assert server["tool_count"] == 0
        assert server["error"] is None
    finally:
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_malformed_tool_entry_disconnects_and_records_error():
    """Unlike a null tools/list result, a non-dict entry in "tools" means the
    server sent garbage — that's a hard failure: record the error and tear
    down rather than leaving a half-registered connection running."""
    client = MCPClient()
    await client.connect("probe", stdio_config(env={
        "PATH": "/usr/bin:/bin", "FAKE_MCP_TOOLS_LIST_MODE": "bad_entry",
    }))
    health = client.get_health()
    server = next(s for s in health["servers"] if s["name"] == "probe")
    assert server["connected"] is False
    assert server["error"]
    # _disconnect_server pops the entry and awaits transport.close(), which
    # blocks until the subprocess is reaped — so by the time connect()
    # returned, nothing was left running.
    assert "probe" not in client._servers


@pytest.mark.asyncio
async def test_non_string_arg_records_error_without_orphaning_process():
    """A non-str entry in args makes create_subprocess_exec raise TypeError,
    not FileNotFoundError/OSError — must still be recorded as a connect
    failure rather than escaping connect() uncaught."""
    client = MCPClient()
    await client.connect("probe", stdio_config(args=["-u", FIXTURE, 5]))
    health = client.get_health()
    server = next(s for s in health["servers"] if s["name"] == "probe")
    assert server["connected"] is False
    assert server["error"]
    assert "probe" not in client._servers
