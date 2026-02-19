"""Workspace path validation — prevents path traversal and sensitive file access."""

from __future__ import annotations

from pathlib import Path

MAX_SEND_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Filenames that should never be served or read via file tools (defense-in-depth).
_SENSITIVE_NAMES = frozenset({".env", "agent.db", "agent.db-shm", "agent.db-wal"})


class PathSecurityError(Exception):
    """Raised when a path fails workspace boundary or sensitivity checks."""


def validate_workspace_path(
    path: str,
    workspace: str | Path,
    *,
    must_exist: bool = False,
    reject_sensitive: bool = False,
) -> Path:
    """Resolve *path* and ensure it stays within *workspace*.

    Args:
        path: User-supplied path (absolute or relative).
        workspace: The workspace root directory.
        must_exist: If True, raise if the resolved file does not exist.
        reject_sensitive: If True, block access to known sensitive filenames.

    Returns:
        The canonicalized ``Path`` within the workspace.

    Raises:
        PathSecurityError: On any violation.
    """
    workspace = Path(workspace).resolve()
    target = Path(path)

    # Treat relative paths as relative to the workspace
    if not target.is_absolute():
        target = workspace / target

    # Resolve symlinks and '..' components
    resolved = target.resolve()

    # Boundary check
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PathSecurityError(
            f"Path escapes workspace boundary: {path}"
        )

    if must_exist and not resolved.exists():
        raise PathSecurityError(f"File does not exist: {path}")

    if reject_sensitive and resolved.name in _SENSITIVE_NAMES:
        raise PathSecurityError(
            f"Access to sensitive file blocked: {resolved.name}"
        )

    return resolved
