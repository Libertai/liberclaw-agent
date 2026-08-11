"""HttpTransport request handling, driven through httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
import time

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


def sse(*events: str) -> httpx.Response:
    return httpx.Response(
        200, headers={"Content-Type": "text/event-stream"}, text="".join(events)
    )


@pytest.mark.asyncio
async def test_sse_response_returns_matching_id():
    def handler(request):
        body = json.loads(request.content.decode())
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 1}})
        return sse(f"data: {payload}\n\n")

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 1}
    await t.close()


@pytest.mark.asyncio
async def test_sse_ignores_comments_and_other_fields():
    def handler(request):
        body = json.loads(request.content.decode())
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 2}})
        return sse(": keep-alive\n\n", "event: message\nid: 7\nretry: 100\n", f"data: {payload}\n\n")

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 2}
    await t.close()


@pytest.mark.asyncio
async def test_sse_joins_multiline_data_into_one_document():
    def handler(request):
        body = json.loads(request.content.decode())
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 3}})
        half = len(payload) // 2
        return sse(f"data: {payload[:half]}\ndata: {payload[half:]}\n\n")

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 3}
    await t.close()


@pytest.mark.asyncio
async def test_sse_skips_interleaved_notification():
    def handler(request):
        body = json.loads(request.content.decode())
        note = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 4}})
        return sse(f"data: {note}\n\n", f"data: {payload}\n\n")

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 4}
    await t.close()


@pytest.mark.asyncio
async def test_interleaved_ping_is_answered():
    posts = []

    def handler(request):
        body = json.loads(request.content.decode())
        posts.append(body)
        if body.get("method") == "tools/list":
            ping = json.dumps({"jsonrpc": "2.0", "id": "p1", "method": "ping"})
            payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 5}})
            return sse(f"data: {ping}\n\n", f"data: {payload}\n\n")
        return httpx.Response(202)

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 5}
    replies = [p for p in posts if p.get("id") == "p1" and "result" in p]
    assert replies and replies[0]["result"] == {}
    await t.close()


@pytest.mark.asyncio
async def test_interleaved_unknown_method_gets_method_not_found():
    posts = []

    def handler(request):
        body = json.loads(request.content.decode())
        posts.append(body)
        if body.get("method") == "tools/list":
            req = json.dumps(
                {"jsonrpc": "2.0", "id": "p2", "method": "sampling/createMessage"}
            )
            payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 7}})
            return sse(f"data: {req}\n\n", f"data: {payload}\n\n")
        return httpx.Response(202)

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 7}
    replies = [p for p in posts if p.get("id") == "p2" and "error" in p]
    assert replies and replies[0]["error"]["code"] == -32601
    await t.close()


@pytest.mark.asyncio
async def test_sse_request_id_collision_with_server_ping():
    # A server-initiated request allocates from its own id space, independent
    # of ours. If ours happens to start at the same value, only the presence
    # of "method" (never on a response) may distinguish it from our answer.
    posts = []

    def handler(request):
        body = json.loads(request.content.decode())
        posts.append(body)
        if body.get("method") == "tools/list":
            ping = json.dumps({"jsonrpc": "2.0", "id": body["id"], "method": "ping"})
            payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 6}})
            return sse(f"data: {ping}\n\n", f"data: {payload}\n\n")
        return httpx.Response(202)

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 6}
    replies = [p for p in posts if "result" in p and p.get("result") == {}]
    assert replies
    await t.close()


@pytest.mark.asyncio
async def test_sse_skips_non_object_payload():
    def handler(request):
        body = json.loads(request.content.decode())
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"v": 8}})
        return sse("data: 5\n\n", f"data: {payload}\n\n")

    t = transport_with(handler)
    assert await t.send_request("tools/list", {}, 5.0) == {"v": 8}
    await t.close()


class _RaisingStream(httpx.AsyncByteStream):
    """A stream whose body read fails, simulating a mid-response network drop."""

    async def __aiter__(self):
        raise httpx.ReadError("boom")
        yield b""  # pragma: no cover


@pytest.mark.asyncio
async def test_error_body_read_failure_raises_mcp_error():
    def handler(request):
        return httpx.Response(500, stream=_RaisingStream())

    t = transport_with(handler)
    with pytest.raises(MCPError, match="500"):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


class _BrokenSseStream(httpx.AsyncByteStream):
    """An SSE body that drops mid-stream, simulating a connection reset."""

    async def __aiter__(self):
        yield b": keep-alive\n\n"
        raise httpx.ReadError("boom")


@pytest.mark.asyncio
async def test_sse_stream_read_error_raises_mcp_error():
    def handler(request):
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=_BrokenSseStream()
        )

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


class _DribblingSseStream(httpx.AsyncByteStream):
    """An SSE body that never ends, emitting only keep-alives."""

    async def __aiter__(self):
        while True:
            await asyncio.sleep(0.05)
            yield b": keep-alive\n\n"


@pytest.mark.asyncio
async def test_sse_timeout_bounds_a_dribbling_stream():
    # httpx's own read timeout resets on every chunk, so a stream that never
    # stops emitting keep-alives would hang forever without the transport's
    # own asyncio.timeout bounding the whole wait.
    def handler(request):
        body = json.loads(request.content.decode())
        if body.get("method") == "notifications/cancelled":
            return httpx.Response(202)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=_DribblingSseStream()
        )

    t = transport_with(handler)
    start = time.monotonic()
    with pytest.raises(MCPError, match="timed out"):
        await t.send_request("tools/list", {}, 0.3)
    elapsed = time.monotonic() - start
    assert 0.2 <= elapsed < 2.0
    await t.close()
