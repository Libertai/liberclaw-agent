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
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version is not None and method != "initialize":
            headers["MCP-Protocol-Version"] = self._protocol_version
        return headers

    async def send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        req_id = next(self._id_counter)
        body = {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}
        response = await self._post(method, body, timeout)

        # _post only returns on status 200; other statuses already raised.
        try:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                data = await self._consume_sse(response, req_id, method, timeout)
            elif "application/json" in content_type:
                await response.aread()
                data = response.json()
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
                    if msg.get("id") == req_id:
                        return msg
                    if msg.get("id") is not None and "method" in msg:
                        await self._reply_to_interleaved_request(msg)
                    # else: a notification, or a reply to someone else's id — ignore
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

            await response.aread()
            text = response.text[:_MAX_ERROR_BODY_LEN]
            await response.aclose()
            raise MCPError(
                f"request '{method}' failed: HTTP {response.status_code}: {text}"
            )

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
