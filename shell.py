"""Persistent bash shell that maintains state across tool calls."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

logger = logging.getLogger(__name__)

# Env vars to strip from the shell environment for security
_SENSITIVE_ENV_VARS = frozenset({
    "AGENT_SECRET_HASH",
    "LIBERTAI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "OWNER_TELEGRAM_ID",
})


class PersistentShell:
    """A long-lived bash process that preserves cwd, env vars, and aliases.

    Each execute() call sends a command to the running bash process and reads
    output until a unique sentinel marker appears in stderr.  If a command
    times out, the bash process is killed and transparently restarted.
    """

    def __init__(self, workspace_path: str) -> None:
        self._workspace_path = workspace_path
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start (or restart) the underlying bash process."""
        await self._spawn()
        self._started = True
        logger.info("Persistent shell started (cwd=%s)", self._workspace_path)

    async def stop(self) -> None:
        """Terminate the bash process."""
        self._started = False
        await self._kill()
        logger.info("Persistent shell stopped")

    async def _spawn(self) -> None:
        """Spawn a fresh bash process with a sanitized environment."""
        # Build a clean env: inherit current env but remove sensitive vars
        env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_VARS}

        self._process = await asyncio.create_subprocess_exec(
            "bash", "--norc", "--noprofile",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workspace_path,
            env=env,
        )

    async def _kill(self) -> None:
        """Kill the current bash process if it's alive."""
        if self._process is None:
            return
        try:
            self._process.kill()
            await self._process.wait()
        except ProcessLookupError:
            pass
        self._process = None

    async def _ensure_alive(self) -> None:
        """Restart the shell if it died unexpectedly."""
        if self._process is None or self._process.returncode is not None:
            logger.warning("Shell process died (rc=%s), restarting",
                           self._process.returncode if self._process else "N/A")
            await self._spawn()

    # ── Command execution ────────────────────────────────────────────────

    async def execute(self, command: str, timeout: int = 60) -> tuple[str, str, int]:
        """Run a command in the persistent shell.

        Returns (stdout, stderr, exit_code).  If the command times out,
        the shell is killed and restarted, and a timeout indicator is returned.
        """
        async with self._lock:
            await self._ensure_alive()
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            assert self._process.stderr is not None

            sentinel = f"__SENTINEL_{uuid.uuid4().hex}__"

            # Build the wrapped command:
            # 1. Run the user's command
            # 2. Capture its exit code
            # 3. Echo a stdout sentinel so we know stdout is done
            # 4. Echo a stderr sentinel with the exit code
            #
            # The stderr sentinel is the authoritative delimiter.
            wrapped = (
                f"{command}\n"
                f"__exit_code__=$?\n"
                f"echo '{sentinel}' >&1\n"
                f"echo '{sentinel}' $__exit_code__ >&2\n"
            )

            self._process.stdin.write(wrapped.encode())
            await self._process.stdin.drain()

            try:
                stdout, stderr, exit_code = await asyncio.wait_for(
                    self._read_until_sentinel(sentinel),
                    timeout=timeout,
                )
                return stdout, stderr, exit_code
            except asyncio.TimeoutError:
                logger.warning("Command timed out after %ds, restarting shell", timeout)
                await self._kill()
                await self._spawn()
                return "", "", -1  # caller handles the timeout message

    async def _read_until_sentinel(self, sentinel: str) -> tuple[str, str, int]:
        """Read stdout and stderr concurrently until the sentinel appears in stderr."""
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        exit_code = 0
        stdout_done = False
        stderr_done = False

        assert self._process is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None

        sentinel_bytes = sentinel.encode()

        async def read_stdout():
            nonlocal stdout_done
            while True:
                line = await self._process.stdout.readline()  # type: ignore[union-attr]
                if not line:
                    # EOF — process died
                    stdout_done = True
                    return
                if sentinel_bytes in line:
                    stdout_done = True
                    return
                stdout_chunks.append(line)

        async def read_stderr():
            nonlocal stderr_done, exit_code
            while True:
                line = await self._process.stderr.readline()  # type: ignore[union-attr]
                if not line:
                    # EOF — process died
                    stderr_done = True
                    return
                if sentinel_bytes in line:
                    # Parse exit code from sentinel line: "SENTINEL <exit_code>"
                    decoded = line.decode("utf-8", errors="replace").strip()
                    parts = decoded.split()
                    if len(parts) >= 2:
                        try:
                            exit_code = int(parts[-1])
                        except ValueError:
                            exit_code = 1
                    stderr_done = True
                    return
                stderr_chunks.append(line)

        # Run both readers concurrently; stderr sentinel is authoritative
        stdout_task = asyncio.create_task(read_stdout())
        stderr_task = asyncio.create_task(read_stderr())

        # Wait for stderr sentinel (the authoritative one)
        await stderr_task

        # Give stdout a brief moment to finish flushing, then cancel
        if not stdout_done:
            try:
                await asyncio.wait_for(stdout_task, timeout=0.5)
            except asyncio.TimeoutError:
                stdout_task.cancel()
                try:
                    await stdout_task
                except asyncio.CancelledError:
                    pass
        else:
            stdout_task.cancel()
            try:
                await stdout_task
            except asyncio.CancelledError:
                pass

        stdout_str = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_str = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        return stdout_str, stderr_str, exit_code
