"""Workspace path validation and command safety — prevents path traversal,
sensitive file access, and dangerous command execution."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

MAX_SEND_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Filenames that should never be served or read via file tools (defense-in-depth).
_SENSITIVE_NAMES = frozenset({".env", "agent.db", "agent.db-shm", "agent.db-wal"})

# ── Bash command safety ────────────────────────────────────────────────

# Dangerous base commands (checked against parsed tokens)
_DANGEROUS_COMMANDS = frozenset({
    "mkfs", "format", "diskpart", "shutdown", "reboot", "poweroff", "halt",
})

# Commands that dump environment variables (expose secrets)
_ENV_DUMP_COMMANDS = frozenset({
    "env", "printenv",
})

# Patterns that are dangerous regardless of parsing
_DANGEROUS_PATTERNS = [
    re.compile(p) for p in [
        r"\brm\s+-[rf]{1,2}\s+/",        # rm -rf /
        r"\brm\s+-[rf]{1,2}\s+~",        # rm -rf ~
        r"\bdd\s+if=",                    # raw disk writes
        r">\s*/dev/sd",                   # device redirection
        r":\(\)\s*\{.*\};\s*:",           # fork bomb
        r"\bsystemctl\s+(stop|disable)\s+baal-agent\b",
        r"\bkill\s+-9\s+1\b",            # kill PID 1
        r"/proc/\d+/environ",            # /proc/<pid>/environ
        r"/proc/self/environ",           # /proc/self/environ
        r"/run/secrets",                 # secrets directory
    ]
]

# Patterns for encoded/obfuscated command bypass attempts
_OBFUSCATION_PATTERNS = [
    re.compile(p) for p in [
        r"\bbase64\s+(-d|--decode)",      # base64 decode pipelines
        r"\beval\s+",                     # eval arbitrary code
        r"\bexec\s+",                     # exec replaces shell
        r"\$\(.*base64.*-d",             # command substitution with base64
        r"`.*base64.*-d",               # backtick with base64
        r"\\x[0-9a-fA-F]{2}",          # hex-encoded chars in commands
    ]
]


def _normalize_command(command: str) -> str:
    """Normalize a command string to defeat trivial escaping tricks.

    Strips trivial quoting tricks like ba''sh or ba""sh, collapses whitespace.
    """
    # Remove empty quotes used to break up command names: rm'' -> rm, ba""sh -> bash
    normalized = re.sub(r"''|\"\"", "", command)
    # Remove single-char quote wrapping: 'r''m' -> rm  (but keep real quoted strings)
    normalized = re.sub(r"'(.)'", r"\1", normalized)
    return normalized


def check_command_safety(command: str) -> str | None:
    """Check if a bash command is safe to execute.

    Uses multiple strategies:
    1. Regex patterns on the raw command (catches common dangerous patterns)
    2. Normalized command check (defeats trivial quoting bypass)
    3. shlex parsing to check individual tokens
    4. Obfuscation detection (base64, eval, hex encoding)

    Returns an error message if blocked, None if safe.
    """
    # Check raw command against dangerous patterns
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"[blocked: command matches safety pattern: {pattern.pattern}]"

    # Normalize and recheck (catches ba''sh -c 'rm -rf /')
    normalized = _normalize_command(command)
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(normalized):
            return f"[blocked: normalized command matches safety pattern: {pattern.pattern}]"

    # Check for obfuscation attempts
    for pattern in _OBFUSCATION_PATTERNS:
        if pattern.search(command):
            return f"[blocked: potential command obfuscation detected: {pattern.pattern}]"

    # Parse with shlex for token-level checks
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Malformed quoting — let it through, bash will error anyway
        tokens = command.split()

    if not tokens:
        return None

    # Check first token and all tokens after pipe/semicolon/&& for dangerous commands
    cmd_positions = {0}  # First token is always a command
    for i, token in enumerate(tokens):
        if token in ("|", ";", "&&", "||"):
            if i + 1 < len(tokens):
                cmd_positions.add(i + 1)

    for pos in cmd_positions:
        if pos >= len(tokens):
            continue
        # Get the base command name (strip path: /usr/bin/rm -> rm)
        base_cmd = tokens[pos].rsplit("/", 1)[-1]
        if base_cmd in _DANGEROUS_COMMANDS:
            return f"[blocked: dangerous command: {base_cmd}]"
        if base_cmd in _ENV_DUMP_COMMANDS:
            return "[blocked: environment variable dump (exposes secrets)]"

    # Check for bare 'set' (dumps vars), 'set |' (piped dump), export -p, declare -x
    for i, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1]
        if base == "set" and (i + 1 >= len(tokens) or tokens[i + 1] == "|"):
            return "[blocked: 'set' without arguments dumps environment variables]"
        if base == "export" and i + 1 < len(tokens) and tokens[i + 1] == "-p":
            return "[blocked: 'export -p' dumps all exported variables]"
        if base == "declare" and i + 1 < len(tokens) and tokens[i + 1] == "-x":
            return "[blocked: 'declare -x' dumps exported variables]"

    # Block reading .env files via common read commands
    read_commands = {"cat", "less", "more", "head", "tail", "source", ".", "nano", "vi", "vim"}
    for i, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1]
        if base in read_commands:
            # Check subsequent tokens for .env patterns
            for j in range(i + 1, min(i + 5, len(tokens))):
                if tokens[j] in ("|", ";", "&&", "||"):
                    break
                if re.search(r"\.env\b", tokens[j]):
                    return "[blocked: reading .env files is not allowed]"

    return None


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
