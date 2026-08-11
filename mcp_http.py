"""Streamable HTTP MCP transport: speaks JSON-RPC over POST to a server URL."""

from __future__ import annotations

import itertools
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
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            raise MCPError("SSE responses are not yet supported")
        if "application/json" not in content_type:
            raise MCPError(
                f"unexpected content-type '{content_type}' for '{method}'"
            )

        data = response.json()
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

    async def _post(self, method: str, body: dict, timeout: float) -> httpx.Response:
        assert self._client is not None
        url = self._url
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                response = await self._client.post(
                    url, json=body, headers=self._headers(method), timeout=timeout
                )
            except httpx.HTTPError as e:
                raise MCPError(f"request '{method}' failed: {e}") from e

            if response.status_code in (301, 302, 303, 307, 308):
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

            text = response.text[:_MAX_ERROR_BODY_LEN]
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
            except Exception:
                pass
        await self._client.aclose()
        self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None
