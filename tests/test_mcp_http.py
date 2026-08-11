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
    holder = {}

    def handler(request):
        holder["response"] = httpx.Response(500, stream=_RaisingStream())
        return holder["response"]

    t = transport_with(handler)
    with pytest.raises(MCPError, match="500"):
        await t.send_request("tools/list", {}, 5.0)
    assert holder["response"].is_closed
    await t.close()


class _BrokenSseStream(httpx.AsyncByteStream):
    """An SSE body that drops mid-stream, simulating a connection reset."""

    async def __aiter__(self):
        yield b": keep-alive\n\n"
        raise httpx.ReadError("boom")


@pytest.mark.asyncio
async def test_sse_stream_read_error_raises_mcp_error():
    holder = {}

    def handler(request):
        holder["response"] = httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=_BrokenSseStream()
        )
        return holder["response"]

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert holder["response"].is_closed
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


@pytest.mark.asyncio
async def test_404_with_session_reinitialises_then_retries():
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        calls.append((body.get("method"), request.headers.get("mcp-session-id")))
        if body.get("method") == "initialize":
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S2"}
            )
        if len([c for c in calls if c[0] == "tools/list"]) == 1:
            return httpx.Response(404, text="session expired")
        return json_rpc_result(request, {"retried": True})

    t = transport_with(handler)
    t.set_reinit_hook(lambda: t.send_request("initialize", {}, 5.0))
    await t.send_request("initialize", {}, 5.0)
    assert await t.send_request("tools/list", {}, 5.0) == {"retried": True}
    assert [c[0] for c in calls].count("initialize") == 2
    await t.close()


@pytest.mark.asyncio
async def test_second_404_fails():
    def handler(request):
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S3"}
            )
        return httpx.Response(404, text="gone")

    t = transport_with(handler)
    t.set_reinit_hook(lambda: t.send_request("initialize", {}, 5.0))
    await t.send_request("initialize", {}, 5.0)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_initialize_itself_does_not_recurse_on_404():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content.decode()).get("method"))
        return httpx.Response(404, text="nope")

    t = transport_with(handler)
    t._session_id = "S4"
    with pytest.raises(MCPError):
        await t.send_request("initialize", {}, 5.0)
    assert calls.count("initialize") == 1
    await t.close()


@pytest.mark.asyncio
async def test_close_sends_delete_with_session():
    seen = []

    def handler(request):
        seen.append((request.method, request.headers.get("mcp-session-id")))
        if request.method == "DELETE":
            return httpx.Response(200)
        return json_rpc_result(
            request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S5"}
        )

    t = transport_with(handler)
    await t.send_request("initialize", {}, 5.0)
    await t.close()
    assert ("DELETE", "S5") in seen


@pytest.mark.asyncio
async def test_404_without_reinit_hook_raises_named_error():
    def handler(request):
        return httpx.Response(404, text="session expired")

    t = transport_with(handler)
    t._session_id = "S6"
    with pytest.raises(MCPError, match="no re-init hook"):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_concurrent_404_recovers_once_and_both_retries_succeed():
    # Both requests observe the same expired session and 404 concurrently.
    # Only one may run the hook; the other must wait for it and retry
    # against the session it produced, rather than hard-failing because the
    # loser's eligibility check read state the winner had already cleared.
    calls = []

    async def handler(request):
        body = json.loads(request.content.decode())
        method = body.get("method")
        sid = request.headers.get("mcp-session-id")
        calls.append((method, sid))
        # Every response yields once immediately: with a synchronous 404
        # neither task ever suspends before the first one reaches recovery
        # and mutates shared state, so the second task's pre-request
        # snapshot would just see the mutated state instead of racing it.
        await asyncio.sleep(0)
        if method == "initialize":
            # Delay the response so the second 404 has time to race in
            # while this recovery is still in flight.
            await asyncio.sleep(0.05)
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S2"}
            )
        if sid != "S2":
            return httpx.Response(404, text="session expired")
        return json_rpc_result(request, {"echo": method})

    t = transport_with(handler)
    t.set_reinit_hook(lambda: t.send_request("initialize", {}, 5.0))
    t._session_id = "S1"

    results = await asyncio.gather(
        t.send_request("tools/list", {}, 5.0),
        t.send_request("tools/call", {}, 5.0),
    )
    assert results[0] == {"echo": "tools/list"}
    assert results[1] == {"echo": "tools/call"}
    assert [c[0] for c in calls].count("initialize") == 1
    await t.close()


@pytest.mark.asyncio
async def test_hook_own_404_does_not_deadlock():
    # The hook's own tools/list refresh 404s on the freshly-issued session.
    # That request runs on the same task already holding the recovery lock;
    # if it tried to re-enter recovery it would deadlock on that lock
    # forever, so this must fail fast (swallowed by the hook) instead.
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        method = body.get("method")
        sid = request.headers.get("mcp-session-id")
        calls.append((method, sid))
        if method == "initialize":
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S2"}
            )
        if sid != "S2":
            return httpx.Response(404, text="session expired")
        if method == "tools/list":
            return httpx.Response(404, text="session expired")
        return json_rpc_result(request, {"echo": method})

    t = transport_with(handler)

    async def hook():
        await t.send_request("initialize", {}, 5.0)
        try:
            await t.send_request("tools/list", {}, 5.0)
        except MCPError:
            pass  # mirrors _reinit_server swallowing a failed tool refresh

    t.set_reinit_hook(hook)
    t._session_id = "S1"

    result = await asyncio.wait_for(t.send_request("tools/call", {}, 5.0), timeout=2.0)
    assert result == {"echo": "tools/call"}
    assert [c[0] for c in calls].count("initialize") == 1
    await t.close()


@pytest.mark.asyncio
async def test_request_entering_during_recovery_does_not_hard_fail():
    # A request that starts fresh WHILE another task's recovery is already
    # in flight — not one that raced in at the same instant — must still
    # see a real session and queue on the lock, not a cleared one that
    # makes it give up immediately.
    calls = []
    late = {}

    async def handler(request):
        body = json.loads(request.content.decode())
        method = body.get("method")
        sid = request.headers.get("mcp-session-id")
        calls.append((method, sid))
        if method == "initialize":
            # Start a brand new request exactly while this recovery
            # handshake is in flight, before responding to it.
            late["task"] = asyncio.create_task(t.send_request("tools/call", {}, 5.0))
            await asyncio.sleep(0.05)
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S2"}
            )
        if sid != "S2":
            return httpx.Response(404, text="session expired")
        return json_rpc_result(request, {"echo": method})

    t = transport_with(handler)
    t.set_reinit_hook(lambda: t.send_request("initialize", {}, 5.0))
    t._session_id = "S1"

    first = await t.send_request("tools/list", {}, 5.0)
    late_result = await late["task"]
    assert first == {"echo": "tools/list"}
    assert late_result == {"echo": "tools/call"}
    assert [c[0] for c in calls].count("initialize") == 1
    await t.close()


@pytest.mark.asyncio
async def test_failed_hook_leaves_session_recoverable():
    # A hook that fails (network drop, 500 on initialize) must not brick the
    # transport: the session and generation must be left exactly as they
    # were, so a later request — once the server is healthy again — can
    # still recover instead of short-circuiting forever.
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        method = body.get("method")
        sid = request.headers.get("mcp-session-id")
        calls.append((method, sid))
        if method == "initialize":
            if [c[0] for c in calls].count("initialize") == 1:
                return httpx.Response(500, text="server hiccup")
            return json_rpc_result(
                request, {"protocolVersion": "2025-06-18"}, headers={"Mcp-Session-Id": "S2"}
            )
        if sid != "S2":
            return httpx.Response(404, text="session expired")
        return json_rpc_result(request, {"echo": method})

    t = transport_with(handler)
    t.set_reinit_hook(lambda: t.send_request("initialize", {}, 5.0))
    t._session_id = "S1"

    generation_before = t._session_generation
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert t._session_id == "S1"
    assert t._session_generation == generation_before

    result = await t.send_request("tools/call", {}, 5.0)
    assert result == {"echo": "tools/call"}
    assert [c[0] for c in calls].count("initialize") == 2
    await t.close()


@pytest.mark.asyncio
async def test_json_response_read_failure_raises_mcp_error():
    holder = {}

    def handler(request):
        holder["response"] = httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=_RaisingStream()
        )
        return holder["response"]

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert holder["response"].is_closed
    await t.close()


@pytest.mark.asyncio
async def test_json_response_malformed_body_raises_mcp_error():
    def handler(request):
        return httpx.Response(200, headers={"Content-Type": "application/json"}, text="not json")

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    await t.close()


@pytest.mark.asyncio
async def test_connected_goes_false_after_transport_level_failure():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    t = transport_with(handler)
    assert t.connected
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert not t.connected
    await t.close()


@pytest.mark.asyncio
async def test_connected_stays_true_after_a_server_error_status():
    """A 4xx/5xx is a server response, not a transport failure — marking
    the transport unhealthy for one would tear down and rebuild a
    perfectly good connection over a single bad status code."""

    def handler(request):
        return httpx.Response(500, text="internal error")

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert t.connected
    await t.close()


@pytest.mark.asyncio
async def test_connected_recovers_after_a_later_successful_response():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return json_rpc_result(request, {"ok": True})

    t = transport_with(handler)
    with pytest.raises(MCPError):
        await t.send_request("tools/list", {}, 5.0)
    assert not t.connected
    await t.send_request("tools/list", {}, 5.0)
    assert t.connected
    await t.close()


@pytest.mark.asyncio
async def test_health_reports_http_transport_and_error():
    from baal_agent.mcp_client import MCPClient

    client = MCPClient()
    await client.connect("api", {"transport": "http", "url": "https://unreachable.invalid/mcp"})
    health = client.get_health()
    server = next(s for s in health["servers"] if s["name"] == "api")
    assert server["transport"] == "http"
    assert server["connected"] is False
    assert server["error"]
