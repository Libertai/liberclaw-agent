"""Execute Python scripts that can call agent tools via JSON-RPC over Unix socket."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid

logger = logging.getLogger(__name__)

MAX_OUTPUT = 30_000

# Helper code prepended to every user script.  Provides call_tool() which
# connects to the JSON-RPC socket and invokes agent tools synchronously.
_HELPER_CODE = '''\
import json, os, socket

def call_tool(name, **kwargs):
    """Call an agent tool and return the result string."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(os.environ["BAAL_TOOL_SOCKET"])
    request = json.dumps({"method": "call_tool", "params": {"name": name, "arguments": kwargs}, "id": 1})
    sock.sendall(request.encode() + b"\\n")
    sock.shutdown(socket.SHUT_WR)
    data = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    sock.close()
    response = json.loads(data)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]

# User code starts below
'''


class CodeExecutor:
    """Runs Python scripts that can call agent tools via JSON-RPC.

    Usage::

        executor = CodeExecutor()
        await executor.start()
        result = await executor.execute("print(call_tool('bash', command='ls'))")
        await executor.stop()
    """

    def __init__(self) -> None:
        self._socket_path: str | None = None
        self._server: asyncio.AbstractServer | None = None

    @property
    def socket_path(self) -> str | None:
        return self._socket_path

    async def start(self) -> None:
        """Start the JSON-RPC Unix socket server."""
        self._socket_path = f"/tmp/baal_tools_{uuid.uuid4().hex}.sock"
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("CodeExecutor socket server started at %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and clean up the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_path and os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass
            self._socket_path = None
        logger.info("CodeExecutor stopped")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single JSON-RPC connection from a child script."""
        try:
            raw = await reader.readline()
            if not raw:
                return
            request = json.loads(raw)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            if method != "call_tool":
                response = {"error": f"unknown method: {method}", "id": req_id}
            else:
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                # Block recursive/dangerous tools from execute_code scripts
                _BLOCKED_TOOLS = {"execute_code", "spawn"}
                if not tool_name:
                    response = {"error": "missing tool name", "id": req_id}
                elif tool_name in _BLOCKED_TOOLS:
                    response = {"error": f"tool '{tool_name}' is not available from execute_code", "id": req_id}
                else:
                    # Import here to avoid circular import at module level
                    from baal_agent.tools import execute_tool

                    try:
                        result = await execute_tool(
                            tool_name, tool_args, pii_redaction=False
                        )
                        response = {"result": result, "id": req_id}
                    except Exception as exc:
                        response = {"error": str(exc), "id": req_id}

            writer.write(json.dumps(response).encode())
            await writer.drain()
        except Exception as exc:
            logger.debug("CodeExecutor connection error: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def execute(self, code: str, timeout: int = 120) -> str:
        """Run a Python script with tool access and return stdout.

        The script receives the ``BAAL_TOOL_SOCKET`` env var and a pre-injected
        ``call_tool(name, **kwargs)`` helper function.

        Args:
            code: Python source code to execute.
            timeout: Wall-clock timeout in seconds (max 300).

        Returns:
            The script's stdout output, truncated to *MAX_OUTPUT* chars.
        """
        if self._socket_path is None:
            return "[error: CodeExecutor not started]"

        timeout = min(timeout, 300)

        # Write combined helper + user code to a temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="baal_exec_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(_HELPER_CODE)
                f.write(code)
                f.write("\n")

            # Strip sensitive env vars from child process
            _sensitive = {"AGENT_SECRET_HASH", "LIBERTAI_API_KEY", "TELEGRAM_BOT_TOKEN", "OWNER_TELEGRAM_ID"}
            env = {k: v for k, v in os.environ.items() if k not in _sensitive}
            env["BAAL_TOOL_SOCKET"] = self._socket_path

            proc = await asyncio.create_subprocess_exec(
                "python3",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return f"[timed out after {timeout}s]"

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr]\n{err}")
            if proc.returncode and proc.returncode != 0:
                parts.append(f"[exit code: {proc.returncode}]")

            result = "\n".join(parts) if parts else "(no output)"

            # Truncate if too long
            if len(result) > MAX_OUTPUT:
                half = MAX_OUTPUT // 2
                result = (
                    result[:half]
                    + f"\n\n... truncated ({len(result)} chars total) ...\n\n"
                    + result[-half:]
                )

            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
