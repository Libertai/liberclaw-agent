"""MCP (Model Context Protocol) client for connecting to external tool servers."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from baal_agent.mcp_http import HttpTransport
from baal_agent.mcp_registry import ToolRegistry
from baal_agent.mcp_stdio import StdioTransport
from baal_agent.mcp_transport import MCPError, Transport

logger = logging.getLogger(__name__)

# Circuit-breaker thresholds. After N consecutive failures from a server,
# we short-circuit subsequent calls for COOLDOWN seconds so a single bad
# server doesn't stall every chat turn (each call has a 60s timeout).
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_COOLDOWN = 60.0

# Tool metadata (name, description) comes from the remote server and lands in
# the system prompt via get_tool_definitions() — the spec requires treating it
# as untrusted unless the server is trusted, so it's bounded before use.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_TOOL_DESCRIPTION_LEN = 1024
_MAX_TOOLS_PER_SERVER = 128
# tools/list page size is server-chosen; this bounds a server that echoes the
# same cursor forever rather than terminating pagination.
_MAX_TOOLS_LIST_PAGES = 50

# Maintenance loop: reconnect backoff and tool-list refresh cadence.
_RECONNECT_BACKOFF_START = 30.0
_RECONNECT_BACKOFF_MAX = 300.0
_TOOLS_TTL = 300.0


@dataclass
class MCPToolInfo:
    """Metadata for a single tool discovered from an MCP server."""

    server_name: str
    original_name: str
    namespaced_name: str  # mcp_{server}_{tool}
    description: str
    input_schema: dict  # JSON Schema for parameters


@dataclass
class MCPToolCallResult:
    """Structured result from an MCP tool call."""

    content: str
    is_error: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class MCPServerConnection:
    """An active connection to an MCP server."""

    name: str
    transport_kind: str  # "stdio" | "http", for health reporting
    transport: Transport
    tools: dict[str, MCPToolInfo] = field(default_factory=dict)
    tools_listed_at: float = 0.0


class MCPClient:
    """Connects to MCP servers and registers their tools."""

    # Protocol versions this client's handshake and tool-call shapes support.
    # Per spec, an unrecognized version in the server's initialize response
    # means the client SHOULD disconnect rather than guess compatibility.
    _SUPPORTED_PROTOCOL_VERSIONS = frozenset(
        {"2025-06-18", "2025-03-26", "2024-11-05"}
    )

    def __init__(self, now: Callable[[], float] | None = None):
        self.registry = ToolRegistry()
        self._configured: dict[str, dict] = {}  # server name -> config
        self._errors: dict[str, str] = {}
        # Circuit breaker state per server.
        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}
        # Injected so tests assert schedule, not real time. The maintenance
        # loop and the circuit breaker share this clock.
        self._now = now or time.monotonic
        # Reconnect backoff state per unconnected server.
        self._retry_at: dict[str, float] = {}
        self._backoff: dict[str, float] = {}
        self._maintain_lock = asyncio.Lock()
        self._maintaining = False
        self._closed = False

    def _circuit_state(self, server_name: str) -> tuple[bool, float]:
        """Return (open, seconds_remaining) for a server."""
        until = self._circuit_open_until.get(server_name, 0.0)
        now = self._now()
        if until > now:
            return True, until - now
        if until:
            # Cooldown elapsed; clear so the next failure counts fresh.
            self._circuit_open_until.pop(server_name, None)
        return False, 0.0

    def _record_success(self, server_name: str) -> None:
        self._failure_counts.pop(server_name, None)
        self._circuit_open_until.pop(server_name, None)

    def _record_failure(self, server_name: str) -> None:
        count = self._failure_counts.get(server_name, 0) + 1
        self._failure_counts[server_name] = count
        if count >= _CIRCUIT_THRESHOLD:
            self._circuit_open_until[server_name] = (
                self._now() + _CIRCUIT_COOLDOWN
            )
            logger.warning(
                "MCP server %r tripped the circuit breaker after %d "
                "consecutive failures; pausing calls for %.0fs",
                server_name,
                count,
                _CIRCUIT_COOLDOWN,
            )

    async def connect(self, name: str, config: dict) -> None:
        """Connect to an MCP server.

        Config keys:
            transport: "stdio" or "http"
            command: command to run (stdio)
            args: list of arguments (stdio)
            env: optional environment variables (stdio)
            url: server URL (http)
        """
        transport = config.get("transport", "stdio")
        self._configured[name] = dict(config)

        if transport == "stdio":
            await self._connect_stdio(name, config)
        elif transport == "http":
            await self._connect_http(name, config)
        else:
            self._errors[name] = f"unknown transport '{transport}'"
            logger.error(f"Unknown MCP transport '{transport}' for server '{name}'")

    async def _connect_stdio(self, name: str, config: dict) -> None:
        """Connect to an MCP server via stdio (subprocess)."""
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            self._errors[name] = "missing command"
            logger.error(f"MCP server '{name}' missing 'command' in config")
            return

        await self._connect_transport(
            name, "stdio", StdioTransport(command, args, env)
        )

    async def _connect_http(self, name: str, config: dict) -> None:
        """Connect to an MCP server via Streamable HTTP."""
        url = config.get("url")

        if not url:
            self._errors[name] = "missing url"
            logger.error(f"MCP server '{name}' missing 'url' in config")
            return

        await self._connect_transport(
            name, "http", HttpTransport(url, config.get("headers"))
        )

    async def _connect_transport(
        self, name: str, kind: str, transport: Transport
    ) -> None:
        """Handshake shared by every transport: initialize, capability check,
        paginated tools/list. Transport-specific setup lives in the caller."""
        try:
            await transport.open()

            conn = MCPServerConnection(
                name=name,
                transport_kind=kind,
                transport=transport,
            )
            # Register before initialize so a failure below still lets
            # _disconnect_server find and close this connection.
            self.registry.register(conn, {})

            # HTTP sessions can expire (404) mid-turn; only HttpTransport knows
            # this hook, so stdio and any other transport skip the wiring.
            if hasattr(transport, "set_reinit_hook"):
                transport.set_reinit_hook(
                    lambda: self._reinit_server(name, transport, conn)
                )

            # Initialize the server. Empty capabilities is deliberate: it's
            # what makes omitting sampling/roots/elicitation conformant —
            # don't declare a capability this client can't service.
            init_result = await transport.send_request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "baal-agent", "version": "1.0.0"},
            })

            version = init_result.get("protocolVersion")
            if version not in self._SUPPORTED_PROTOCOL_VERSIONS:
                raise MCPError(f"unsupported protocol version {version}")

            # Send initialized notification (no response expected)
            await transport.send_notification("notifications/initialized", {})
        except Exception as e:
            self._errors[name] = str(e)
            logger.error(f"MCP server '{name}' connection failed: {e}")
            await self._disconnect_server(name)
            return

        # No tools capability: stay connected (already registered with zero
        # tools above) rather than treat it as a handshake failure.
        if "tools" not in init_result.get("capabilities", {}):
            self._errors[name] = str(MCPError("server does not offer tools"))
            logger.warning(f"MCP server '{name}' initialized without a tools capability")
            return

        # Discover tools, paginating until the server stops returning a
        # cursor. A request failure here leaves the connection open with zero
        # tools rather than tearing down a server that initialized fine.
        try:
            try:
                tools = await self._list_tools_for(conn)
            except MCPError:
                logger.warning(f"MCP server '{name}' tools/list returned nothing")
                return

            self.registry.register(conn, tools)
            conn.tools_listed_at = self._now()

            logger.info(
                f"MCP server '{name}' connected: {len(tools)} tools discovered"
            )
            self._errors.pop(name, None)
        except Exception as e:
            self._errors[name] = str(e)
            logger.error(f"MCP server '{name}' connection failed: {e}")
            await self._disconnect_server(name)

    async def _list_all_tools(
        self, name: str, transport: Transport
    ) -> dict[str, MCPToolInfo]:
        """Paginate tools/list until the server stops returning a cursor.

        Raises MCPError if any page fails — the caller decides how a partial
        listing should be handled. Shared by the initial connect and by
        session-recovery re-init, so both discover tools the same way.
        """
        tools: dict[str, MCPToolInfo] = {}
        cursor = None
        for _ in range(_MAX_TOOLS_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            tools_result = await transport.send_request("tools/list", params)
            tools.update(self._tools_from_list_result(name, tools_result))
            cursor = tools_result.get("nextCursor")
            # Stop paginating once the cap is already met — a server that
            # never stops advertising a cursor would otherwise force up to
            # _MAX_TOOLS_LIST_PAGES round-trips before truncation.
            if not cursor or len(tools) >= _MAX_TOOLS_PER_SERVER:
                break

        if len(tools) > _MAX_TOOLS_PER_SERVER:
            logger.warning(
                f"MCP server '{name}' offered more than {_MAX_TOOLS_PER_SERVER} "
                f"tools; keeping the first {_MAX_TOOLS_PER_SERVER}"
            )
            tools = dict(list(tools.items())[:_MAX_TOOLS_PER_SERVER])
        return tools

    async def _list_tools_for(
        self, conn: MCPServerConnection
    ) -> dict[str, MCPToolInfo]:
        """Paginated tools/list for an existing connection.

        Thin wrapper over _list_all_tools so connect() and the maintenance
        refresh share one pagination path, and tests can stub the
        connection-level call without touching (name, transport) plumbing.
        """
        return await self._list_all_tools(conn.name, conn.transport)

    def _next_backoff(self, current: float) -> float:
        return min(max(current * 2, _RECONNECT_BACKOFF_START), _RECONNECT_BACKOFF_MAX)

    async def maintain(self) -> None:
        """Reconnect unreachable servers with backoff, refresh tool lists
        past their TTL.

        This is the first background caller into MCP request paths — every
        prior call came from a chat turn. self._maintaining is checked
        before the lock so an overlapping tick (loop tick racing another
        loop tick, or a manual call) returns immediately rather than
        queueing behind one already running.
        """
        if self._maintaining:
            return
        async with self._maintain_lock:
            if self._maintaining:
                return
            self._maintaining = True
            try:
                await self._reconnect_unreachable()
                if self._closed:
                    return
                await self._refresh_stale_tools()
            finally:
                self._maintaining = False

    async def _reconnect_unreachable(self) -> None:
        for name, config in list(self._configured.items()):
            if self._closed:
                return
            if name in self.registry.servers:
                continue
            is_open, _ = self._circuit_state(name)
            if is_open:
                continue
            if self._now() < self._retry_at.get(name, 0.0):
                continue

            # Unconditional: a prior failed attempt can leave a transport
            # half-open (registered before initialize, per _connect_transport's
            # own teardown), and skipping this orphans that httpx.AsyncClient
            # or subprocess on every retry cycle.
            await self._disconnect_server(name)
            if self._closed:
                return
            try:
                await self.connect(name, config)
            except Exception as e:
                logger.warning("MCP server %r reconnect failed: %s", name, e)
            if self._closed:
                return

            if name in self.registry.servers:
                self._retry_at.pop(name, None)
                self._backoff.pop(name, None)
            else:
                backoff = self._next_backoff(self._backoff.get(name, 0.0))
                self._backoff[name] = backoff
                self._retry_at[name] = self._now() + backoff

    async def _refresh_stale_tools(self) -> None:
        for name, conn in list(self.registry.servers.items()):
            if self._closed:
                return
            if not conn.transport.connected:
                continue
            if self._now() - conn.tools_listed_at <= _TOOLS_TTL:
                continue
            is_open, _ = self._circuit_state(name)
            if is_open:
                continue

            try:
                tools = await self._list_tools_for(conn)
            except Exception as e:
                # Keep the current tools and leave tools_listed_at untouched
                # so the next tick retries — one blip must not empty the
                # model's tool set mid-conversation.
                logger.warning("MCP server %r tool refresh failed: %s", name, e)
                continue
            if self._closed:
                return

            if self.registry.replace_tools(conn, tools):
                conn.tools_listed_at = self._now()

    async def _reinit_server(
        self, name: str, transport: Transport, conn: MCPServerConnection
    ) -> None:
        """Re-run the handshake for HttpTransport's 404 session recovery, then
        refresh tools — a restarted server may advertise a different set, so
        without this the tool list stays stale until the maintenance TTL.
        """
        init_result = await transport.send_request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "baal-agent", "version": "1.0.0"},
        })
        version = init_result.get("protocolVersion")
        if version not in self._SUPPORTED_PROTOCOL_VERSIONS:
            raise MCPError(f"unsupported protocol version {version}")
        await transport.send_notification("notifications/initialized", {})

        if "tools" not in init_result.get("capabilities", {}):
            return

        try:
            tools = await self._list_all_tools(name, transport)
        except MCPError:
            logger.warning(f"MCP server '{name}' tools/list returned nothing during reinit")
            return
        self.registry.replace_tools(conn, tools)

    def _tools_from_list_result(
        self, server_name: str, result: dict
    ) -> dict[str, MCPToolInfo]:
        """Build namespaced tools from one tools/list page.

        Drops names failing the identifier pattern and truncates descriptions
        — untrusted metadata from the server, bounded before it reaches the
        prompt. The per-server tool count cap is enforced by the caller, which
        accumulates across pages.
        """
        tools: dict[str, MCPToolInfo] = {}
        dropped = False
        for tool_def in result.get("tools", []):
            tool_name = tool_def.get("name", "")
            if not _TOOL_NAME_RE.match(tool_name):
                dropped = True
                continue
            namespaced = f"mcp_{server_name}_{tool_name}"
            tools[namespaced] = MCPToolInfo(
                server_name=server_name,
                original_name=tool_name,
                namespaced_name=namespaced,
                description=tool_def.get("description", "")[:_MAX_TOOL_DESCRIPTION_LEN],
                input_schema=tool_def.get("inputSchema", {}),
            )
        if dropped:
            logger.warning(
                f"MCP server '{server_name}' offered tool name(s) rejected by "
                f"the naming pattern; dropped"
            )
        return tools

    async def disconnect_all(self) -> None:
        """Disconnect from all servers.

        Sets _closed first so a maintain() call still in flight (e.g. not
        reached by the maintenance task's own cancel-and-await, such as a
        direct call) bails at its next check instead of reconnecting or
        re-registering a server after teardown.
        """
        self._closed = True
        for name in list(self.registry.servers.keys()):
            await self._disconnect_server(name)

    async def _disconnect_server(self, name: str) -> None:
        """Disconnect from a single server."""
        conn = self.registry.servers.get(name)
        if conn is None:
            return

        self.registry.remove(name)
        await conn.transport.close()

        logger.info(f"MCP server '{name}' disconnected")

    def get_health(self) -> dict:
        """Return lightweight MCP health for /info and runtime diagnostics."""
        servers = []
        all_names = sorted(set(self._configured) | set(self.registry.servers))
        for name in all_names:
            conn = self.registry.servers.get(name)
            servers.append({
                "name": name,
                "transport": conn.transport_kind if conn else self._configured.get(name, {}).get("transport", "unknown"),
                "connected": conn.transport.connected if conn else False,
                "tool_count": len(conn.tools) if conn else 0,
                "pending_requests": conn.transport.pending_count if conn else 0,
                "error": self._errors.get(name),
            })

        return {
            "enabled": bool(self._configured or self.registry.servers),
            "server_count": len(all_names),
            "connected_count": sum(1 for server in servers if server["connected"]),
            "tool_count": len(self.registry.all_tools()),
            "servers": servers,
        }

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-format tool definitions for all discovered MCP tools."""
        defs = []
        for info in self.registry.all_tools().values():
            # Convert MCP JSON Schema to OpenAI function-calling format
            parameters = dict(info.input_schema) if info.input_schema else {
                "type": "object",
                "properties": {},
            }
            # Ensure the schema has the required "type" field
            if "type" not in parameters:
                parameters["type"] = "object"

            defs.append({
                "type": "function",
                "function": {
                    "name": info.namespaced_name,
                    "description": (
                        f"[MCP: {info.server_name}] {info.description}"
                    ),
                    "parameters": parameters,
                },
            })
        return defs

    async def call_tool_result(
        self,
        namespaced_name: str,
        arguments: dict,
        *,
        image_callback=None,
    ) -> MCPToolCallResult:
        """Call an MCP tool and return text plus structured metadata."""
        info = self.registry.get(namespaced_name)
        base_metadata = {
            "provider": "mcp",
            "server": info.server_name if info else None,
            "original_name": info.original_name if info else None,
            "namespaced_name": namespaced_name,
            "content_types": [],
            "mcp_is_error": False,
        }
        if info is None:
            # A mid-turn tool refresh (maintenance loop) can retire a tool the
            # model was already offered — distinguish that from a name it
            # never had, which points at a model hallucination instead.
            retired_by = _server_name_from_namespaced(namespaced_name, self.registry.servers)
            if retired_by is not None:
                return MCPToolCallResult(
                    content=f"[error: tool no longer offered by server '{retired_by}']",
                    is_error=True,
                    metadata={**base_metadata, "server": retired_by},
                )
            return MCPToolCallResult(
                content=f"[error: unknown MCP tool '{namespaced_name}']",
                is_error=True,
                metadata=base_metadata,
            )

        conn = self.registry.servers.get(info.server_name)
        if conn is None:
            return MCPToolCallResult(
                content=f"[error: MCP server '{info.server_name}' not connected]",
                is_error=True,
                metadata=base_metadata,
            )

        # Fail fast when the breaker is open. Each MCP call is bounded by a
        # 60s timeout — without this, a flaky server adds 60s of latency to
        # every chat turn until it recovers.
        is_open, remaining = self._circuit_state(info.server_name)
        if is_open:
            return MCPToolCallResult(
                content=(
                    f"[error: MCP server '{info.server_name}' circuit open; "
                    f"retry in {remaining:.0f}s]"
                ),
                is_error=True,
                metadata={**base_metadata, "circuit_open": True},
            )

        try:
            result = await conn.transport.send_request("tools/call", {
                "name": info.original_name,
                "arguments": arguments,
            }, timeout=60.0)

            # MCP tool results have a "content" array with text/image blocks
            content_blocks = result.get("content", [])
            content_types = [
                block.get("type", "unknown")
                for block in content_blocks
                if isinstance(block, dict)
            ]
            metadata = {
                **base_metadata,
                "content_types": content_types,
                "mcp_is_error": bool(result.get("isError", False)),
            }
            if not content_blocks:
                return MCPToolCallResult(
                    content="(empty result)",
                    is_error=metadata["mcp_is_error"],
                    metadata=metadata,
                )

            # Forward image blocks to the model via the optional callback so
            # vision-capable models see them as image_url content. Without this
            # MCP image tools (screenshot servers, OCR servers, etc.) were
            # invisible to the model.
            image_blocks = _mcp_image_blocks(
                content_blocks, info.server_name, info.original_name
            )
            if image_blocks and image_callback is not None:
                try:
                    image_callback(image_blocks)
                except Exception:
                    logger.warning(
                        "MCP image callback raised; continuing without images",
                        exc_info=True,
                    )

            parts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    mime = block.get("mimeType", "unknown")
                    if image_blocks:
                        parts.append(f"[Image forwarded to model: {mime}]")
                    else:
                        parts.append(f"[image: {mime}]")
                elif block.get("type") == "resource":
                    uri = block.get("resource", {}).get("uri", "")
                    parts.append(f"[resource: {uri}]")
                else:
                    parts.append(json.dumps(block))

            if image_blocks:
                metadata["image_count"] = len(image_blocks) // 2

            # Server-side `isError=True` is a protocol-level error from the
            # tool itself (model called it with bad args, etc.). It does not
            # mean the transport is unhealthy, so it shouldn't trip the
            # breaker — only transport/timeout failures should.
            if metadata["mcp_is_error"]:
                # Leave failure counter unchanged.
                pass
            else:
                self._record_success(info.server_name)
            return MCPToolCallResult(
                content="\n".join(parts),
                is_error=metadata["mcp_is_error"],
                metadata=metadata,
            )

        except MCPError as e:
            self._record_failure(info.server_name)
            return MCPToolCallResult(
                content=f"[error: MCP tool call failed: {e}]",
                is_error=True,
                metadata=base_metadata,
            )
        except Exception as e:
            self._record_failure(info.server_name)
            logger.error(f"MCP tool '{namespaced_name}' call error: {e}")
            return MCPToolCallResult(
                content=f"[error: MCP tool call error: {e}]",
                is_error=True,
                metadata=base_metadata,
            )

    async def call_tool(self, namespaced_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result as a string."""
        result = await self.call_tool_result(namespaced_name, arguments)
        return result.content


def _server_name_from_namespaced(namespaced_name: str, servers: dict) -> str | None:
    """Recover the owning server name from `mcp_{server}_{tool}` when the
    tool itself isn't registered (e.g. retired by a mid-turn refresh)."""
    for name in servers:
        if namespaced_name.startswith(f"mcp_{name}_"):
            return name
    return None


def _mcp_image_blocks(content_blocks: list, server_name: str, tool_name: str) -> list:
    """Convert MCP image blocks to OpenAI image_url content blocks.

    Returns an empty list if no images are present.
    """
    blocks: list[dict] = []
    for index, block in enumerate(content_blocks):
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        data = block.get("data") or block.get("base64")
        mime = block.get("mimeType") or "image/png"
        if not data:
            continue
        # The MCP spec allows raw bytes or already-encoded base64; we re-encode
        # bytes to keep the data URI consistent for the downstream image
        # callback, which expects "data:<mime>;base64,<payload>".
        if isinstance(data, (bytes, bytearray)):
            payload = base64.b64encode(bytes(data)).decode("ascii")
        else:
            payload = str(data)
        data_uri = f"data:{mime};base64,{payload}"
        blocks.append({
            "type": "text",
            "text": f"[Image #{index + 1} from mcp_{server_name}_{tool_name}]",
        })
        blocks.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    return blocks
