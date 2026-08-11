"""Stdio MCP transport: spawns the server as a subprocess, speaks JSON-RPC over its stdin/stdout."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os

from baal_agent.mcp_transport import MCPError, Transport

logger = logging.getLogger(__name__)

# Vars withheld from the spawned subprocess's env: deployer secrets, tokens, and
# the MCP config itself (which may embed other servers' commands/envs).
_SENSITIVE_ENV = {
    "AGENT_SECRET_HASH",
    "LIBERTAI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "OWNER_TELEGRAM_ID",
    "MCP_SERVERS",
    "MCP_SERVERS_B64",
}


class StdioTransport(Transport):
    """MCP transport over a subprocess's stdin/stdout."""

    def __init__(self, command: str, args: list[str], env: dict | None = None):
        self._command = command
        self._args = args
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._id_counter = itertools.count(1)

    async def open(self) -> None:
        proc_env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV}
        proc_env.update(self._env or {})

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._command, *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
        except FileNotFoundError:
            raise MCPError(f"command '{self._command}' not found") from None
        except OSError as e:
            raise MCPError(str(e)) from e

        self._reader_task = asyncio.create_task(
            self._reader(), name=f"mcp-reader-{self._command}"
        )

    async def _reader(self) -> None:
        """Background task that reads JSON-RPC responses from stdout."""
        assert self._process and self._process.stdout
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    # Process closed stdout
                    logger.warning(f"MCP server '{self._command}' closed stdout")
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    future = self._pending.pop(msg_id)
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
            logger.error(f"MCP reader for '{self._command}' crashed: {e}")
        finally:
            # Resolve any pending futures with errors
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        MCPError(f"Server '{self._command}' disconnected")
                    )
            self._pending.clear()

    async def send_request(
        self, method: str, params: dict, timeout: float = 30.0,
    ) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        if not self._process or not self._process.stdin:
            raise MCPError(f"MCP server '{self._command}' has no active stdin")

        req_id = next(self._id_counter)
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        try:
            data = json.dumps(msg) + "\n"
            self._process.stdin.write(data.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._pending.pop(req_id, None)
            logger.error(f"MCP server '{self._command}' write failed: {e}")
            raise MCPError(f"write failed: {e}") from e

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(req_id, None)
            logger.warning(f"MCP request '{method}' to '{self._command}' timed out")
            raise MCPError(f"request '{method}' timed out")
        except MCPError as e:
            logger.warning(f"MCP request '{method}' to '{self._command}' failed: {e}")
            raise

        # A JSON-RPC response with a null/missing "result" is a malformed
        # reply, not success — callers rely on send_request never returning
        # None so they can treat a dict result as a given.
        if result is None:
            raise MCPError(f"request '{method}' returned no result")
        return result

    async def send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(msg) + "\n"
            self._process.stdin.write(data.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    async def close(self) -> None:
        """Tear down. Safe to call on a transport that never opened."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"Error stopping MCP server '{self._command}': {e}")

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pending_count(self) -> int:
        return len(self._pending)
