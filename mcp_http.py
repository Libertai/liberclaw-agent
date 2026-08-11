"""Streamable HTTP MCP transport: speaks JSON-RPC over POST to a server URL."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from urllib.parse import urljoin, urlsplit

import httpx

from baal_agent.mcp_transport import MCPError, Transport

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 3
_MAX_ERROR_BODY_LEN = 500


class _SessionExpiredError(MCPError):
    """404 for a request that carried Mcp-Session-Id: the session expired server-side."""


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    return parts.scheme, parts.hostname or "", parts.port


async def _sse_messages(lines):
    """Yield decoded JSON messages from an SSE line stream.

    A line opening with ':' is a comment — SSE keep-alives are near-universal on
    long-lived MCP streams and must not be read as malformed input. A multi-line
    ``data:`` event is ONE document joined by newlines, not several. Events
    dispatch on the blank line.
    """
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line:
            if data:
                try:
                    yield json.loads("\n".join(data))
                except json.JSONDecodeError:
                    logger.warning("MCP SSE event was not valid JSON; skipping")
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
        # event/id/retry carry no payload for a tools-only client
    if data:
        try:
            yield json.loads("\n".join(data))
        except json.JSONDecodeError:
            logger.warning("MCP SSE stream ended mid-event; skipping")


class HttpTransport(Transport):
    """MCP transport over HTTP POST (Streamable HTTP)."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self._url = url
        self._user_headers = dict(headers or {})
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._id_counter = itertools.count(1)
        # MCPClient installs this via set_reinit_hook: re-run initialize (with
        # protocolVersion validation) + notifications/initialized. Recovery
        # raises rather than guessing at a handshake if nothing is installed.
        self._reinit_hook = None
        self._reinit_lock = asyncio.Lock()
        # Bumped once a re-init succeeds. Lets a second caller racing on the
        # same expired session detect that another caller already
        # recovered, so it doesn't re-initialise twice.
        self._session_generation = 0
        # Task currently running the recovery hook, while _reinit_lock is
        # held. A 404 raised by a request the hook itself makes (same task)
        # must bail instead of trying to re-acquire this non-reentrant lock.
        self._recovering_task: asyncio.Task | None = None

    async def open(self) -> None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            follow_redirects=False,
        )
        try:
            self._client = client
        except Exception:
            await client.aclose()
            raise

    def _headers(self, method: str) -> httpx.Headers:
        # httpx.Headers merges case-insensitively, so a user-supplied
        # "content-type" doesn't slip past the overwrite below.
        headers = httpx.Headers(self._user_headers)
        # Transport-owned headers always win: a user-supplied Content-Type
        # would otherwise break every request.
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        # Never on initialize, even if self._session_id still holds a stale
        # value: the spec requires a new session's InitializeRequest to
        # carry no session id at all.
        if self._session_id is not None and method != "initialize":
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version is not None and method != "initialize":
            headers["MCP-Protocol-Version"] = self._protocol_version
        return headers

    def set_reinit_hook(self, hook) -> None:
        """Install the 404-recovery handshake.

        `hook` is an async callable taking no arguments, re-running
        `initialize` (with protocolVersion validation) +
        `notifications/initialized`, plus whatever else the caller needs
        (MCPClient also refreshes the tool registry) — the transport has no
        handshake logic of its own, so a 404 recovers only if this is set.
        """
        self._reinit_hook = hook

    async def send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        req_id = next(self._id_counter)
        body = {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}
        generation = self._session_generation
        # Captured now, not read live in the except clause below: another
        # task's recovery can clear self._session_id while this request is
        # in flight, and that must not change what THIS request is eligible
        # for — it observed a real session when it was sent.
        had_session = self._session_id is not None
        try:
            return await self._send_and_parse(method, body, req_id, timeout)
        except _SessionExpiredError:
            # initialize excluded so a 404 there can never re-enter recovery.
            # A 404 from a request the recovery hook itself makes runs on
            # this same task while _recovering_task holds the lock — must
            # bail rather than re-acquire a lock it's already holding.
            if (
                method == "initialize"
                or not had_session
                or asyncio.current_task() is self._recovering_task
            ):
                raise
            await self._recover_session(generation)
            return await self._send_and_parse(method, body, req_id, timeout)

    async def _recover_session(self, generation: int) -> None:
        """Re-run the handshake once; leave the old session in place until it
        succeeds.

        Two concurrent requests can both 404 on the same expired session. The
        lock serialises recovery; the generation check lets whichever caller
        loses the race see that another caller already re-initialised, so it
        retries against the new session instead of re-initialising again.
        Not clearing `_session_id` up front (it's excluded from `initialize`
        regardless, in `_headers`) means a third request that starts fresh
        while this is in flight still observes a real session and queues on
        the lock too, instead of seeing none and giving up immediately. It
        also means a failing hook (network drop, 500 on `initialize`) leaves
        the transport exactly as able to recover as before the attempt —
        nothing to restore.
        """
        async with self._reinit_lock:
            if generation != self._session_generation:
                return
            if self._reinit_hook is None:
                raise MCPError("session expired and no re-init hook is installed")
            self._recovering_task = asyncio.current_task()
            try:
                await self._reinit_hook()
                self._session_generation += 1
            finally:
                self._recovering_task = None

    async def _send_and_parse(
        self, method: str, body: dict, req_id: int, timeout: float
    ) -> dict:
        response = await self._post(method, body, timeout)

        # _post only returns on status 200; other statuses already raised.
        try:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                data = await self._consume_sse(response, req_id, method, timeout)
            elif "application/json" in content_type:
                try:
                    await response.aread()
                    data = response.json()
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    raise MCPError(f"request '{method}' failed to read response: {e}") from e
            else:
                raise MCPError(
                    f"unexpected content-type '{content_type}' for '{method}'"
                )
        finally:
            await response.aclose()

        if "error" in data:
            raise MCPError(data["error"].get("message", "unknown error"))
        result = data.get("result")
        if result is None:
            raise MCPError(f"request '{method}' returned no result")

        if method == "initialize":
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
            self._protocol_version = result.get("protocolVersion")

        return result

    async def _consume_sse(
        self, response: httpx.Response, req_id: int, method: str, timeout: float
    ) -> dict:
        """Read SSE events until one answers `req_id`, replying to any
        interleaved server->client request along the way.

        httpx's read timeout applies per chunk, so a keep-alive comment every
        few seconds would reset it forever; only this outer timeout bounds the
        whole wait.
        """
        try:
            async with asyncio.timeout(timeout):
                async for msg in _sse_messages(response.aiter_lines()):
                    if not isinstance(msg, dict):
                        # _sse_messages guarantees valid JSON, not a JSON object.
                        continue
                    # A response never carries "method" — only a server->client
                    # request/notification does. JSON-RPC ids are per-requestor,
                    # so a bare id match doesn't identify a response: the server's
                    # own id space can collide with ours (e.g. both start at 1).
                    if "method" in msg:
                        if msg.get("id") is not None:
                            await self._reply_to_interleaved_request(msg)
                        continue
                    if msg.get("id") == req_id:
                        return msg
                    # else: a reply to someone else's id — ignore
        except TimeoutError:
            logger.warning(f"MCP request '{method}' timed out waiting on SSE stream")
            # Best effort: a failure to notify the server must never mask the
            # timeout itself.
            try:
                await self.send_notification(
                    "notifications/cancelled", {"requestId": req_id}
                )
            except Exception:
                logger.warning(
                    f"MCP server failed to receive cancellation for '{method}'"
                )
            raise MCPError(f"request '{method}' timed out") from None
        except httpx.HTTPError as e:
            raise MCPError(f"SSE stream for '{method}' failed: {e}") from e

        raise MCPError(f"SSE stream for '{method}' ended without a matching response")

    async def _reply_to_interleaved_request(self, msg: dict) -> None:
        """Answer a server->client request arriving on the SSE stream.

        `ping` sits outside capability negotiation and must be answered
        promptly; with no GET stream open, this POST is the only channel back
        to the server. Anything else gets a "method not found" error.
        """
        msg_id = msg["id"]
        if msg.get("method") == "ping":
            reply = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {msg.get('method')}",
                },
            }
        try:
            await self._client.post(
                self._url, json=reply, headers=self._headers("notification")
            )
        except httpx.HTTPError as e:
            logger.warning(f"MCP interleaved reply failed: {e}")

    async def _post(self, method: str, body: dict, timeout: float) -> httpx.Response:
        """POST `body`, following same-origin redirects.

        Returns an open, unread response (`stream=True`) — the caller reads
        and closes it, since an SSE body must be consumed incrementally
        rather than buffered whole.
        """
        assert self._client is not None
        url = self._url
        for _ in range(_MAX_REDIRECTS + 1):
            request = self._client.build_request(
                "POST", url, json=body, headers=self._headers(method), timeout=timeout
            )
            try:
                response = await self._client.send(request, stream=True)
            except httpx.HTTPError as e:
                raise MCPError(f"request '{method}' failed: {e}") from e

            if response.status_code in (301, 302, 303, 307, 308):
                await response.aclose()
                location = response.headers.get("location")
                if not location:
                    raise MCPError(f"redirect from '{method}' had no Location header")
                target = urljoin(url, location)
                # A cross-origin redirect would carry the Authorization header
                # (and session id) to a host the caller never authorized.
                if _origin(target) != _origin(url):
                    target_host = urlsplit(target).hostname
                    raise MCPError(
                        f"refusing cross-origin redirect to '{target_host}'"
                    )
                url = target
                continue

            if response.status_code == 200:
                return response

            try:
                await response.aread()
            except httpx.HTTPError as e:
                raise MCPError(f"request '{method}' failed: HTTP {response.status_code}: {e}") from e
            finally:
                await response.aclose()
            text = response.text[:_MAX_ERROR_BODY_LEN]
            message = f"request '{method}' failed: HTTP {response.status_code}: {text}"
            if response.status_code == 404:
                raise _SessionExpiredError(message)
            raise MCPError(message)

        raise MCPError(f"request '{method}' exceeded {_MAX_REDIRECTS} redirects")

    async def send_notification(self, method: str, params: dict) -> None:
        if self._client is None:
            return
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            response = await self._client.post(
                self._url, json=body, headers=self._headers(method)
            )
        except httpx.HTTPError as e:
            logger.warning(f"MCP notification '{method}' failed: {e}")
            return
        if response.status_code != 202:
            logger.warning(
                f"MCP notification '{method}' got unexpected status "
                f"{response.status_code}"
            )

    async def close(self) -> None:
        if self._client is None:
            return
        if self._session_id is not None:
            # Best-effort session teardown: the response (including a
            # 404/405 for a server that never tracked the session) is
            # never used, so any failure here is safe to swallow.
            try:
                await self._client.delete(
                    self._url, headers=self._headers("delete")
                )
            except (httpx.HTTPError, ValueError):
                pass
        await self._client.aclose()
        self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None
