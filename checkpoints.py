"""Git-based workspace checkpoints for safe rollback."""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Git-based workspace checkpoints for safe rollback.

    Uses a separate GIT_DIR (.baal-history) so the checkpoint repo
    does not conflict with any user git repos in the workspace.
    """

    def __init__(self, workspace_path: str):
        self.workspace = workspace_path
        self.git_dir = os.path.join(workspace_path, ".baal-history")
        self._initialized = False

    def _env(self) -> dict[str, str]:
        """Return env vars that point git at our hidden repo."""
        env = os.environ.copy()
        env["GIT_DIR"] = self.git_dir
        env["GIT_WORK_TREE"] = self.workspace
        return env

    async def _run(self, *args: str, check: bool = True) -> tuple[str, str, int]:
        """Run a git command with our custom GIT_DIR/GIT_WORK_TREE.

        Returns (stdout, stderr, returncode).
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
                cwd=self.workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()
            code = proc.returncode or 0
            if check and code != 0:
                logger.warning("git %s failed (code %d): %s", " ".join(args), code, err)
            return out, err, code
        except FileNotFoundError:
            return "", "git is not installed", 127
        except asyncio.TimeoutError:
            return "", "git command timed out", -1

    async def init(self) -> None:
        """Initialize the checkpoint git repo if it doesn't exist."""
        if self._initialized:
            return

        # Check if git is available
        out, err, code = await self._run("--version", check=False)
        if code != 0:
            logger.warning("git not available for checkpoints: %s", err)
            return

        if not os.path.isdir(self.git_dir):
            await self._run("init")
            await self._run("config", "user.name", "baal-agent")
            await self._run("config", "user.email", "agent@baal")

            # Create .gitignore for sensitive/transient files
            gitignore_path = os.path.join(self.workspace, ".baal-history-gitignore")
            with open(gitignore_path, "w") as f:
                f.write("agent.db\nagent.db-journal\nagent.db-wal\nagent.db-shm\n.env\n.baal-history/\n")
            # Tell git to use this as the exclude file
            await self._run("config", "core.excludesFile", gitignore_path)

            # Initial commit so we have a baseline
            await self._run("add", "-A")
            await self._run("commit", "-m", "initial checkpoint", "--allow-empty")

        self._initialized = True

    async def create(self, message: str) -> str:
        """Create a checkpoint. Returns checkpoint ID (short SHA) or 'no changes'."""
        await self.init()

        await self._run("add", "-A")

        # Check if there's anything to commit
        _, _, code = await self._run("diff", "--cached", "--quiet", check=False)
        if code == 0:
            return "no changes"

        out, err, code = await self._run("commit", "-m", message)
        if code != 0:
            return f"error: {err}"

        # Get the short SHA
        sha_out, _, _ = await self._run("rev-parse", "--short", "HEAD")
        return sha_out

    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """List recent checkpoints. Returns [{id, message, timestamp}]."""
        await self.init()

        out, err, code = await self._run(
            "log", f"--pretty=format:%h\t%s\t%ci", f"-n{limit}",
            check=False,
        )
        if code != 0 or not out:
            return []

        checkpoints = []
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                checkpoints.append({
                    "id": parts[0],
                    "message": parts[1],
                    "timestamp": parts[2],
                })
        return checkpoints

    async def restore(self, checkpoint_id: str) -> str:
        """Restore workspace to a checkpoint state. Returns summary of changes."""
        await self.init()

        # Validate the checkpoint ID exists
        _, err, code = await self._run("cat-file", "-t", checkpoint_id, check=False)
        if code != 0:
            return f"error: checkpoint '{checkpoint_id}' not found"

        # Get diff summary before restoring
        diff_out, _, _ = await self._run("diff", checkpoint_id, "--stat", check=False)

        # Restore files from the checkpoint (does not move HEAD)
        _, err, code = await self._run("checkout", checkpoint_id, "--", ".")
        if code != 0:
            return f"error restoring: {err}"

        # Stage the restored state so future diffs are clean
        await self._run("add", "-A")
        await self._run("commit", "-m", f"restored to {checkpoint_id}", "--allow-empty")

        if diff_out:
            return f"Restored to checkpoint {checkpoint_id}.\n\nChanges reverted:\n{diff_out}"
        return f"Restored to checkpoint {checkpoint_id} (no file differences)."

    async def diff(self, checkpoint_id: str) -> str:
        """Show what changed since a checkpoint."""
        await self.init()

        # Validate the checkpoint ID exists
        _, err, code = await self._run("cat-file", "-t", checkpoint_id, check=False)
        if code != 0:
            return f"error: checkpoint '{checkpoint_id}' not found"

        # Stage current state so we can diff against it
        await self._run("add", "-A")

        out, _, code = await self._run("diff", checkpoint_id, "--stat", check=False)
        if code != 0 or not out:
            return "No changes since checkpoint."
        return out
