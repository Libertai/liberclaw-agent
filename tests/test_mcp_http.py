"""HttpTransport request handling, driven through httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from baal_agent.mcp_http import HttpTransport
from baal_agent.mcp_transport import MCPError

URL = "https://server.example/mcp"


def json_rpc_result(request, result, headers=None):
    body = json.loads(request.content.decode())
    return httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
        headers=headers or {},
    )


def transport_with(handler) -> HttpTransport:
    t = HttpTransport(URL, headers={"Authorization": "Bearer tok"})
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


@pytest.mark.asyncio
async def test_json_response_returns_result():
    def handler(request):
        return json_rpc_result(request, {"ok": True})

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"ok": True}
    await t.close()


@pytest.mark.asyncio
async def test_user_headers_sent_and_transport_headers_win():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return json_rpc_result(request, {})

    t = HttpTransport(URL, headers={"Authorization": "Bearer tok", "Accept": "text/plain"})
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await t.send_request("tools/list", {}, 5.0)
    assert seen["authorization"] == "Bearer tok"
    assert "application/json" in seen["accept"]
    assert "text/event-stream" in seen["accept"]
    await t.close()


@pytest.mark.asyncio
async def test_protocol_version_absent_on_initialize_present_after():
    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        return json_rpc_result(request, {"protocolVersion": "2025-06-18"})

    t = transport_with(handler)
    await t.send_request("initialize", {}, 5.0)
    await t.send_request("tools/list", {}, 5.0)
    assert "mcp-protocol-version" not in seen[0]
    assert seen[1]["mcp-protocol-version"] == "2025-06-18"
    await t.close()


@pytest.mark.asyncio
async def test_session_id_captured_and_echoed():
    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        return json_rpc_result(
            request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S1"}
        )

    t = transport_with(handler)
    await t.send_request("initialize", {}, 5.0)
    await t.send_request("tools/list", {}, 5.0)
    assert "mcp-session-id" not in seen[0]
    assert seen[1]["mcp-session-id"] == "S1"
    await t.close()


@pytest.mark.asyncio
async def test_json_rpc_error_raises_mcp_error():
    def handler(request):
        body = json.loads(request.content.decode())
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"],
            "error": {"code": -32601, "message": "no such method"},
        })

    t = transport_with(handler)
    with pytest.raises(MCPError, match="no such method"):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_401_body_is_surfaced():
    def handler(request):
        return httpx.Response(401, text="token expired, rotate it")

    t = transport_with(handler)
    with pytest.raises(MCPError, match="token expired"):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_notification_accepts_202():
    def handler(request):
        return httpx.Response(202)

    t = transport_with(handler)
    await t.send_notification("notifications/initialized", {})
    await t.close()


@pytest.mark.asyncio
async def test_cross_origin_redirect_refused():
    def handler(request):
        if request.url.host == "server.example":
            return httpx.Response(307, headers={"Location": "https://evil.example/mcp"})
        return json_rpc_result(request, {})

    t = transport_with(handler)
    with pytest.raises(MCPError, match="evil.example"):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_same_origin_redirect_followed():
    def handler(request):
        if request.url.path == "/mcp":
            return httpx.Response(307, headers={"Location": "https://server.example/mcp/"})
        return json_rpc_result(request, {"followed": True})

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"followed": True}
    await t.close()
