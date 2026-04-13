"""MCP (Model Context Protocol) client for connecting to external tool servers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# JSON-RPC request ID counter
_next_id = 0


def _get_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


@dataclass
class MCPToolInfo:
    """Metadata for a single tool discovered from an MCP server."""

    server_name: str
    original_name: str
    namespaced_name: str  # mcp_{server}_{tool}
    description: str
    input_schema: dict  # JSON Schema for parameters


@dataclass
class MCPServerConnection:
    """An active connection to an MCP server."""

    name: str
    transport: str  # "stdio" or "http"
    process: asyncio.subprocess.Process | None = None
    tools: dict[str, MCPToolInfo] = field(default_factory=dict)
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _pending: dict[int, asyncio.Future] = field(default_factory=dict)
    _reader_task: asyncio.Task | None = None
    url: str | None = None  # for http transport


class MCPClient:
    """Connects to MCP servers and registers their tools."""

    def __init__(self):
        self._servers: dict[str, MCPServerConnection] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # namespaced_name -> info

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

        if transport == "stdio":
            await self._connect_stdio(name, config)
        elif transport == "http":
            logger.warning(f"MCP HTTP transport not yet implemented for server '{name}'")
        else:
            logger.error(f"Unknown MCP transport '{transport}' for server '{name}'")

    async def _connect_stdio(self, name: str, config: dict) -> None:
        """Connect to an MCP server via stdio (subprocess)."""
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            logger.error(f"MCP server '{name}' missing 'command' in config")
            return

        try:
            import os
            _sensitive = {"AGENT_SECRET_HASH", "LIBERTAI_API_KEY", "TELEGRAM_BOT_TOKEN", "OWNER_TELEGRAM_ID"}
            proc_env = {k: v for k, v in os.environ.items() if k not in _sensitive}
            proc_env.update(env or {})

            process = await asyncio.create_subprocess_exec(
                command, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )

            conn = MCPServerConnection(
                name=name,
                transport="stdio",
                process=process,
            )
            self._servers[name] = conn

            # Start background reader for responses
            conn._reader_task = asyncio.create_task(
                self._stdio_reader(conn),
                name=f"mcp-reader-{name}",
            )

            # Initialize the server
            init_result = await self._send_request(conn, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "baal-agent", "version": "1.0.0"},
            })

            if init_result is None:
                logger.error(f"MCP server '{name}' initialization failed")
                await self._disconnect_server(name)
                return

            # Send initialized notification (no response expected)
            await self._send_notification(conn, "notifications/initialized", {})

            # Discover tools
            tools_result = await self._send_request(conn, "tools/list", {})
            if tools_result is None:
                logger.warning(f"MCP server '{name}' tools/list returned nothing")
                return

            tools_list = tools_result.get("tools", [])
            for tool_def in tools_list:
                tool_name = tool_def.get("name", "")
                namespaced = f"mcp_{name}_{tool_name}"
                info = MCPToolInfo(
                    server_name=name,
                    original_name=tool_name,
                    namespaced_name=namespaced,
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                )
                conn.tools[namespaced] = info
                self._tools[namespaced] = info

            logger.info(
                f"MCP server '{name}' connected: {len(conn.tools)} tools discovered"
            )

        except FileNotFoundError:
            logger.error(f"MCP server '{name}': command '{command}' not found")
        except Exception as e:
            logger.error(f"MCP server '{name}' connection failed: {e}")
            await self._disconnect_server(name)

    async def _stdio_reader(self, conn: MCPServerConnection) -> None:
        """Background task that reads JSON-RPC responses from stdout."""
        assert conn.process and conn.process.stdout
        try:
            while True:
                line = await conn.process.stdout.readline()
                if not line:
                    # Process closed stdout
                    logger.warning(f"MCP server '{conn.name}' closed stdout")
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in conn._pending:
                    future = conn._pending.pop(msg_id)
                    if not future.done():
                        if "error" in msg:
                            future.set_exception(
                                MCPError(msg["error"].get("message", "unknown error"))
                            )
                        else:
                            future.set_result(msg.get("result"))
                # Notifications and other messages are ignored for now

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MCP reader for '{conn.name}' crashed: {e}")
        finally:
            # Resolve any pending futures with errors
            for future in conn._pending.values():
                if not future.done():
                    future.set_exception(
                        MCPError(f"Server '{conn.name}' disconnected")
                    )
            conn._pending.clear()

    async def _send_request(
        self, conn: MCPServerConnection, method: str, params: dict,
        timeout: float = 30.0,
    ) -> dict | None:
        """Send a JSON-RPC request and wait for the response."""
        if not conn.process or not conn.process.stdin:
            return None

        req_id = _get_id()
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        conn._pending[req_id] = future

        try:
            data = json.dumps(msg) + "\n"
            conn.process.stdin.write(data.encode("utf-8"))
            await conn.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            conn._pending.pop(req_id, None)
            logger.error(f"MCP server '{conn.name}' write failed: {e}")
            return None

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            conn._pending.pop(req_id, None)
            logger.warning(f"MCP request '{method}' to '{conn.name}' timed out")
            return None
        except MCPError as e:
            logger.warning(f"MCP request '{method}' to '{conn.name}' failed: {e}")
            return None

    async def _send_notification(
        self, conn: MCPServerConnection, method: str, params: dict,
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not conn.process or not conn.process.stdin:
            return

        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(msg) + "\n"
            conn.process.stdin.write(data.encode("utf-8"))
            await conn.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._servers.keys()):
            await self._disconnect_server(name)
        self._tools.clear()

    async def _disconnect_server(self, name: str) -> None:
        """Disconnect from a single server."""
        conn = self._servers.pop(name, None)
        if conn is None:
            return

        # Remove tools from the global index
        for tool_name in list(conn.tools.keys()):
            self._tools.pop(tool_name, None)

        # Cancel reader task
        if conn._reader_task and not conn._reader_task.done():
            conn._reader_task.cancel()
            try:
                await conn._reader_task
            except asyncio.CancelledError:
                pass

        # Terminate process
        if conn.process:
            try:
                conn.process.stdin.close() if conn.process.stdin else None
                conn.process.terminate()
                try:
                    await asyncio.wait_for(conn.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    conn.process.kill()
                    await conn.process.wait()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"Error stopping MCP server '{name}': {e}")

        logger.info(f"MCP server '{name}' disconnected")

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-format tool definitions for all discovered MCP tools."""
        defs = []
        for info in self._tools.values():
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

    async def call_tool(self, namespaced_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result as a string."""
        info = self._tools.get(namespaced_name)
        if info is None:
            return f"[error: unknown MCP tool '{namespaced_name}']"

        conn = self._servers.get(info.server_name)
        if conn is None:
            return f"[error: MCP server '{info.server_name}' not connected]"

        try:
            result = await self._send_request(conn, "tools/call", {
                "name": info.original_name,
                "arguments": arguments,
            }, timeout=60.0)

            if result is None:
                return "[error: MCP tool call returned no result]"

            # MCP tool results have a "content" array with text/image blocks
            content_blocks = result.get("content", [])
            if not content_blocks:
                return "(empty result)"

            parts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                elif block.get("type") == "resource":
                    uri = block.get("resource", {}).get("uri", "")
                    parts.append(f"[resource: {uri}]")
                else:
                    parts.append(json.dumps(block))

            return "\n".join(parts)

        except MCPError as e:
            return f"[error: MCP tool call failed: {e}]"
        except Exception as e:
            logger.error(f"MCP tool '{namespaced_name}' call error: {e}")
            return f"[error: MCP tool call error: {e}]"


class MCPError(Exception):
    """Error from an MCP server."""
    pass
