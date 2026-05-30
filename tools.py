"""Tool definitions and executors for agent VMs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import time as _time
import uuid
from dataclasses import dataclass, field as _field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from baal_agent.image_utils import (
    build_image_content_blocks,
    encode_bytes_to_data_uri,
    is_image,
)
from baal_agent.checkpoints import CheckpointManager
from baal_agent.security import (
    MAX_SEND_FILE_SIZE,
    PathSecurityError,
    check_command_safety,
    validate_workspace_path,
)
from baal_agent.code_executor import CodeExecutor
from baal_agent.shell import PersistentShell

MAX_TOOL_OUTPUT = 30_000
MAX_WEB_CONTENT = 50_000

_IMAGE_AWARE_TOOLS = {"read_file", "read_pdf", "web_fetch", "browser"}
_MUTATING_TOOLS = {
    "bash",
    "apply_patch",
    "write_file",
    "edit_file",
    "multi_edit",
    "remember_fact",
    "send_file",
    "generate_image",
    "todo",
    "execute_code",
    "checkpoint",
    "process",
    "run_format",
    # The next three shell out to user commands that routinely have side
    # effects (coverage caches, generated artifacts, ephemeral test DBs).
    # Running them in parallel with concurrent write_file/edit_file/apply_patch
    # would race; treat them as mutating so run_tool_calls_ordered serializes.
    "run_tests",
    "run_lint",
    "run_typecheck",
    "spawn",
    # The browser keeps stateful navigation (cookies, current page) across
    # calls; serialize it so concurrent actions don't race on the shared page.
    "browser",
}
_SHELL_TOOLS = {
    "bash",
    "execute_code",
    "git_blame",
    "git_diff",
    "git_show",
    "git_status",
    "process",
    "run_format",
    "run_lint",
    "run_tests",
    "run_typecheck",
}


def _split_tool_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class ToolPolicy:
    """Declarative tool policy enforced before execution."""

    mode: str = "full-auto"
    allowlist: frozenset[str] = frozenset()
    denylist: frozenset[str] = frozenset()

    @classmethod
    def from_strings(
        cls,
        *,
        mode: str = "full-auto",
        allowlist: str = "",
        denylist: str = "",
    ) -> "ToolPolicy":
        normalized = (mode or "full-auto").strip()
        return cls(
            mode=normalized,
            allowlist=frozenset(_split_tool_list(allowlist)),
            denylist=frozenset(_split_tool_list(denylist)),
        )

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "allowlist": sorted(self.allowlist),
            "denylist": sorted(self.denylist),
        }

    def check(self, name: str) -> tuple[bool, str | None]:
        if name in self.denylist:
            return False, f"tool '{name}' is denied by policy"
        if self.allowlist and name not in self.allowlist:
            return False, f"tool '{name}' is not in the policy allowlist"
        if self.mode == "full-auto":
            return True, None
        if self.mode == "auto-read":
            if is_mutating_tool(name):
                return False, f"tool '{name}' is mutating and policy mode is auto-read"
            return True, None
        if self.mode == "locked-down":
            return False, f"tool '{name}' is blocked by locked-down policy"
        if self.mode == "ask-before-write":
            if is_mutating_tool(name):
                return False, f"tool '{name}' requires approval in ask-before-write mode"
            return True, None
        if self.mode == "ask-before-shell":
            if name in _SHELL_TOOLS:
                return False, f"tool '{name}' requires approval in ask-before-shell mode"
            return True, None
        return False, f"unknown tool policy mode '{self.mode}'"


# ── Subagent role policies ──────────────────────────────────────────
#
# Per-role tool subsets. Each role declares the allowlist of tools its
# subagent may invoke. `worker` / `default` mean "inherit the global policy"
# and stay None. The policies layer on top of the parent's policy: the
# stricter of (parent, role) wins via _intersect_policies.

# Read-only tools available to every constrained role.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "read_pdf",
    "list_dir",
    "glob",
    "grep",
    "search_history",
    "search_memory",
})

_SUBAGENT_ROLE_POLICIES: dict[str, "ToolPolicy"] = {
    "explorer": ToolPolicy(
        mode="auto-read",
        allowlist=_READ_ONLY_TOOLS,
    ),
    "reviewer": ToolPolicy(
        mode="auto-read",
        allowlist=_READ_ONLY_TOOLS | frozenset({"git_diff", "git_show", "git_blame", "git_status"}),
    ),
    "verifier": ToolPolicy(
        mode="ask-before-write",
        allowlist=_READ_ONLY_TOOLS
        | frozenset({
            "run_tests",
            "run_lint",
            "run_typecheck",
            "git_diff",
            "git_status",
        }),
    ),
    "researcher": ToolPolicy(
        mode="auto-read",
        allowlist=_READ_ONLY_TOOLS | frozenset({"web_fetch", "web_search"}),
    ),
}


def subagent_role_policy(role: str | None) -> "ToolPolicy | None":
    """Return the per-role ToolPolicy, or None for unrestricted roles.

    `default` and `worker` get no extra restriction; the parent agent's
    global policy still applies as usual. Other roles return a role-scoped
    allowlist that the dispatcher intersects with the global policy.
    """
    if not role:
        return None
    return _SUBAGENT_ROLE_POLICIES.get(role.strip().lower())


def intersect_policies(
    base: "ToolPolicy | None", overlay: "ToolPolicy | None"
) -> "ToolPolicy | None":
    """Return a policy at least as strict as both inputs.

    Used to stack a subagent role policy on top of the global agent
    policy: any allowlist on either side narrows the resulting set; the
    stricter mode wins.
    """
    if base is None and overlay is None:
        return None
    if base is None:
        return overlay
    if overlay is None:
        return base
    # Mode precedence (most → least restrictive).
    order = [
        "locked-down",
        "auto-read",
        "ask-before-write",
        "ask-before-shell",
        "full-auto",
    ]
    base_rank = order.index(base.mode) if base.mode in order else len(order)
    overlay_rank = order.index(overlay.mode) if overlay.mode in order else len(order)
    mode = base.mode if base_rank <= overlay_rank else overlay.mode

    # Allowlist intersection: if either side has an allowlist, the result
    # is the intersection of all declared allowlists.
    if base.allowlist and overlay.allowlist:
        allowlist = base.allowlist & overlay.allowlist
    else:
        allowlist = base.allowlist or overlay.allowlist
    return ToolPolicy(
        mode=mode,
        allowlist=allowlist,
        denylist=base.denylist | overlay.denylist,
    )


@dataclass
class ToolExecutionContext:
    """Turn-scoped state shared across tool calls."""

    read_hashes: dict[str, str] = _field(default_factory=dict)
    # The chat_id of the current conversation. Empty string means "unscoped",
    # in which case memory writes default to global / unscoped storage.
    chat_id: str = ""
    # The active ToolPolicy for this turn. Carried through ToolExecutionContext
    # so that nested dispatchers (notably execute_code's sandbox bridge) can
    # re-apply policy on inner tool calls instead of running unchecked.
    policy: "ToolPolicy | None" = None


@dataclass
class ToolResult:
    """Structured metadata for a tool execution."""

    name: str
    content: str
    is_error: bool = False
    duration_ms: int = 0
    truncated: bool = False
    metadata: dict = _field(default_factory=dict)
    artifacts: list[dict] = _field(default_factory=list)

    def to_event(self) -> dict:
        preview = self.content[:2_000]
        if len(self.content) > 2_000:
            preview += "\n... preview truncated ..."
        return {
            "type": "tool_result",
            "name": self.name,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "content": preview,
        }


def _is_error_result(content: str) -> bool:
    return content.startswith("[error:") or content.startswith("Error executing ")


def _is_truncated_result(content: str) -> bool:
    lowered = content.lower()
    return "[large output saved:" in content or "... truncated" in lowered or "omitted" in lowered


def tool_metadata() -> list[dict]:
    """Return public metadata for built-in tools."""
    metadata = []
    for tool in TOOL_DEFINITIONS:
        name = tool["function"]["name"]
        available, reason = _tool_available(name)
        metadata.append({
            "name": name,
            "available": available,
            "unavailable_reason": reason,
            "mutating": is_mutating_tool(name),
            "image_aware": name in _IMAGE_AWARE_TOOLS,
        })
    return metadata

# ── Workspace configuration ──────────────────────────────────────────

_workspace_path: str | None = None
_db = None  # AgentDatabase instance, set via configure_tools
_shell: PersistentShell | None = None
_checkpoint_mgr: CheckpointManager | None = None
_code_executor: CodeExecutor | None = None
_mcp_client = None  # MCPClient instance, set via start_mcp
_inference = None  # InferenceClient instance, for LLM-powered tools
_model: str = ""  # Model name, for LLM-powered tools


def configure_tools(workspace_path: str, db=None, inference=None, model: str = "") -> None:
    """Set the workspace root and optional database for tool boundary checks."""
    global _workspace_path, _db, _inference, _model
    _workspace_path = workspace_path
    _db = db
    if inference is not None:
        _inference = inference
    if model:
        _model = model


async def start_shell() -> None:
    """Create and start the persistent shell for bash tool calls."""
    global _shell
    if _workspace_path is None:
        raise RuntimeError("configure_tools() must be called before start_shell()")
    _shell = PersistentShell(_workspace_path)
    await _shell.start()


async def shutdown_shell() -> None:
    """Stop the persistent shell. Safe to call even if not started."""
    global _shell
    if _shell is not None:
        await _shell.stop()
        _shell = None


async def start_code_executor() -> None:
    """Create and start the code executor for execute_code tool calls."""
    global _code_executor
    _code_executor = CodeExecutor()
    await _code_executor.start()


async def shutdown_code_executor() -> None:
    """Stop the code executor. Safe to call even if not started."""
    global _code_executor
    if _code_executor is not None:
        await _code_executor.stop()
        _code_executor = None


async def start_mcp(mcp_servers_json: str) -> None:
    """Parse MCP server config and connect to all configured servers."""
    global _mcp_client
    if not mcp_servers_json.strip():
        return
    try:
        servers = json.loads(mcp_servers_json)
    except json.JSONDecodeError as e:
        import logging as _logging
        _logging.getLogger(__name__).error(f"Invalid mcp_servers JSON: {e}")
        return
    if not isinstance(servers, list) or not servers:
        return

    from baal_agent.mcp_client import MCPClient
    _mcp_client = MCPClient()
    for srv in servers:
        name = srv.get("name")
        if not name:
            continue
        try:
            await _mcp_client.connect(name, srv)
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).error(f"MCP server '{name}' connect failed: {e}")


async def shutdown_mcp() -> None:
    """Disconnect from all MCP servers. Safe to call even if not started."""
    global _mcp_client
    if _mcp_client is not None:
        await _mcp_client.disconnect_all()
        _mcp_client = None


def get_mcp_health() -> dict:
    """Return health for the active MCP client, if configured."""
    if _mcp_client is None:
        return {
            "enabled": False,
            "server_count": 0,
            "connected_count": 0,
            "tool_count": 0,
            "servers": [],
        }
    return _mcp_client.get_health()

# ── Bash safety guards ────────────────────────────────────────────────
# Legacy BASH_DENY_PATTERNS kept for backward compatibility with tests.
# The actual safety check is now done by check_command_safety() in security.py
# which uses shlex parsing + normalization for harder-to-bypass protection.
BASH_DENY_PATTERNS = []

# ── Tool definitions ──────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60, max 300).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents with line numbers. For image files (png, jpg, gif, webp, bmp), returns the image visually so you can see it. Binary files are detected and return metadata with inspection hints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_many_files",
            "description": (
                "Read up to 20 text files in one call, returning each file with "
                "line numbers. Use this for efficient multi-file code inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to read, relative to the workspace or absolute within it.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from for every file (1-based).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines per file.",
                    },
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Read a PDF file. In text mode (default), extracts text from pages — fast and lightweight, "
                "good for most documents. In image mode, renders pages as images for visual analysis — "
                "use when layout, diagrams, or tables matter. Use this instead of read_file for PDF files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the PDF file.",
                    },
                    "pages": {
                        "type": "string",
                        "description": 'Page(s) to read. Examples: "1", "1-3", "2,5,8". Defaults to "1". Max 3 pages per call in image mode, 20 in text mode.',
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["text", "image"],
                        "description": 'Reading mode. "text" (default) extracts text content. "image" renders pages visually.',
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Find and replace an exact string in a text file. By default the "
                "old string must appear exactly once. Set replace_all=true to "
                "replace every occurrence. Optionally pass expected_hash "
                "(SHA-256 of the current file) to reject stale edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences. Defaults to false.",
                    },
                    "expected_hash": {
                        "type": "string",
                        "description": "Optional SHA-256 hash the file must match before editing.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": (
                "Apply multiple exact string replacements. Each edit follows "
                "edit_file safety rules: the file must have been read this turn "
                "or the edit must include expected_hash."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                                "expected_hash": {"type": "string"},
                            },
                            "required": ["path", "old_string", "new_string"],
                        },
                        "description": "Ordered edit_file operations to apply.",
                    },
                },
                "required": ["edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff patch to files in the workspace. The patch "
                "is checked before it is applied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified diff text accepted by git apply.",
                    },
                },
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory with [dir] and [file] prefixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files by filename pattern inside the workspace. Results are "
                "sorted by modification time, newest first. Dotfiles and dot-directories "
                "(.git, .env, etc.) are not matched by '*' or '**/*' — match them "
                "explicitly with patterns like '.*' or '**/.git/**'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern such as '*.py' or '**/*.md'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search within. Defaults to workspace root.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 100, max 500).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search text files with ripgrep inside the workspace. Use this "
                "instead of bash for code/text search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern passed to ripgrep.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search. Defaults to workspace root.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Result mode. Defaults to content.",
                    },
                    "type_filter": {
                        "type": "string",
                        "description": "Optional ripgrep file type filter, e.g. 'py', 'ts', 'md'.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "If false, run case-insensitive search.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum output lines to return (default 200, max 1000).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Return concise git status for the workspace repository.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Return git diff for workspace changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional path to limit the diff."},
                    "staged": {"type": "boolean", "description": "If true, show staged diff."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_show",
            "description": "Show a git object or commit with optional path filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rev": {"type": "string", "description": "Revision or object to show. Defaults to HEAD."},
                    "path": {"type": "string", "description": "Optional path to limit output."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_blame",
            "description": "Show git blame for a file, optionally limited to a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to blame."},
                    "start": {"type": "integer", "description": "Optional starting line."},
                    "end": {"type": "integer", "description": "Optional ending line."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a test command in the workspace and return output plus exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Test command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_lint",
            "description": "Run a lint command in the workspace and return output plus exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Lint command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_typecheck",
            "description": "Run a typecheck command in the workspace and return output plus exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Typecheck command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_format",
            "description": "Run a formatting command in the workspace and return output plus exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Format command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its text content (HTML tags stripped).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (http or https).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser",
            "description": (
                "Drive a headless Chromium browser to interact with live pages "
                "(JavaScript-rendered sites, forms, login walls). Actions: "
                "'goto' loads a URL; 'extract' returns the rendered page text; "
                "'click' clicks an element by CSS selector; 'type' types text into "
                "an input by CSS selector; 'screenshot' captures the current page as "
                "an image; 'back' navigates to the previous page. The browser keeps "
                "its state (cookies, current page) across calls within a session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["goto", "extract", "click", "type", "screenshot", "back"],
                        "description": "The browser action to perform.",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (required for 'goto'; http or https only).",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for 'click' and 'type'.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (required for 'type').",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_file",
            "description": "Send a file from the workspace to the user via Telegram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to workspace or absolute within workspace).",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption to send with the file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search your past conversation history using full-text search. "
                "Use this to recall what was discussed about a topic, find details "
                "from previous conversations, or check if something was mentioned before. "
                "Set summarize=true to get a synthesized answer instead of raw snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            'Search query. Supports FTS5 syntax: words, "exact phrases", '
                            "OR, NOT, prefix*."
                        ),
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Optional: limit search to a specific conversation.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 20, max 50).",
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "If true, use the LLM to synthesize a coherent answer from the search results instead of returning raw snippets.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Store a compact typed memory record for durable future use. "
                "Use only for stable facts, repo conventions, decisions, or "
                "test commands that will remain useful later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "Memory category, e.g. user_fact, repo_convention, test_command, decision.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Compact declarative memory content.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source label. Defaults to agent.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional structured metadata.",
                    },
                },
                "required": ["kind", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search typed memory records stored by remember_fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for. Empty lists recent records.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Optional kind/category filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 20, max 100).",
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Whether archived records should be returned.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using LibertAI Search. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-10, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text prompt using LibertAI's image generation API. "
                "Max size 1024x1024. Dimensions must be multiples of 16. "
                "Steps: 8 default (fast), use 14 for text readability or high quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate.",
                    },
                    "size": {
                        "type": "string",
                        "description": 'Image dimensions as "WxH", e.g. "1024x1024" (default). Max 1024 per side. Must be multiples of 16.',
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Generation steps. Default 8 (fast). Use 14 for text readability or high quality output.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Manage a structured task list. Use this to plan and track multi-step work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "update", "complete", "delete"],
                        "description": "Action to perform.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title (for add).",
                    },
                    "id": {
                        "type": "integer",
                        "description": "Task ID (for update/complete/delete).",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status (for update).",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority (for add/update).",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes (for add/update).",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute a Python script that can call agent tools programmatically. "
                "Tool results from the script do NOT enter the conversation context, "
                "making this ideal for multi-step operations that would otherwise consume "
                "context window. The script's stdout is returned. Use call_tool(name, **kwargs) "
                "to invoke any agent tool (bash, read_file, write_file, edit_file, list_dir, "
                "web_fetch, web_search, etc)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. A call_tool(name, **kwargs) function is "
                            "pre-injected for invoking agent tools. Print results you want returned."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120, max 300).",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkpoint",
            "description": "Create, list, restore, or diff workspace checkpoints. Checkpoints are lightweight git snapshots for safe rollback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "restore", "diff"],
                        "description": "Action to perform.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Checkpoint message (required for create).",
                    },
                    "id": {
                        "type": "string",
                        "description": "Checkpoint ID/SHA (required for restore/diff).",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process",
            "description": "Manage long-running background processes. Start commands, check status, read output, or kill processes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "list", "poll", "kill"],
                        "description": "Action to perform.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command to start (for start action).",
                    },
                    "id": {
                        "type": "string",
                        "description": "Process ID (for poll/kill).",
                    },
                },
                "required": ["action"],
            },
        },
    },
]

# Spawn tool — added dynamically in main.py (not available to subagents)
SPAWN_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "spawn",
        "description": (
            "Spawn a background subagent to work on a task asynchronously. "
            "The subagent runs with its own tool set (no further spawning) and "
            "can be given a persona to specialize its behavior. Results are "
            "delivered as pending messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description for the subagent.",
                },
                "label": {
                    "type": "string",
                    "description": "Short label for the task (used in result notification).",
                },
                "persona": {
                    "type": "string",
                    "description": "Optional system prompt override for the subagent. Gives it a specialized role.",
                },
                "role": {
                    "type": "string",
                    "enum": ["default", "explorer", "worker", "reviewer", "verifier", "researcher"],
                    "description": (
                        "Typed role for the subagent. Use explorer for read-only codebase "
                        "inspection, worker for bounded implementation, reviewer for code "
                        "review, verifier for tests/checks, and researcher for synthesis."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Wall-clock timeout in seconds (default 300, max 600).",
                },
            },
            "required": ["task"],
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────

_ERROR_PATTERNS = re.compile(
    r"\b(?:error|Error|ERROR|failed|FAILED|warning|WARNING|traceback|Traceback)\b"
)


def _spill_tool_output(text: str, source: str = "") -> str | None:
    """Persist oversized tool output to workspace/tool-results and return a preview."""
    if not _workspace_path:
        return None
    try:
        workspace = Path(_workspace_path)
        out_dir = workspace / "tool-results"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "-", source or "tool").strip("-")
        filename = f"{safe_source or 'tool'}-{uuid.uuid4().hex[:12]}.txt"
        path = out_dir / filename
        path.write_text(text)
        rel = path.relative_to(workspace)
        line_count = text.count("\n") + (1 if text else 0)
        preview_budget = min(MAX_TOOL_OUTPUT - 500, 8_000)
        head = preview_budget * 2 // 3
        tail = preview_budget - head
        preview = text[:head]
        if len(text) > preview_budget:
            preview += (
                f"\n\n... full output saved to {rel} "
                f"({len(text):,} chars, {line_count:,} lines) ...\n\n"
                + text[-tail:]
            )
        return (
            f"[large output saved: {rel} "
            f"({len(text):,} chars, {line_count:,} lines)]\n\n"
            f"{preview}"
        )
    except Exception:
        return None


def _truncate(text: str, source: str = "") -> str:
    """Context-aware truncation of tool output.

    For bash output: keeps first 20% and last 30% of *lines*, plus any lines
    from the middle that match common error/warning patterns.

    For all other output: keeps first 40% and last 20% of characters (weighted
    toward the beginning which is usually most relevant).

    Always stays within MAX_TOOL_OUTPUT.
    """
    if len(text) <= MAX_TOOL_OUTPUT:
        return text

    spilled = _spill_tool_output(text, source)
    if spilled is not None:
        return spilled

    total_chars = len(text)
    # Reserve space for the truncation notice
    notice_budget = 80  # enough for the notice line
    budget = MAX_TOOL_OUTPUT - notice_budget

    if source == "bash":
        # Line-based truncation with error-line preservation
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        head_count = max(1, int(total_lines * 0.20))
        tail_count = max(1, int(total_lines * 0.30))

        # Prevent overlap
        if head_count + tail_count >= total_lines:
            # Just do char-based fallback
            head_chars = budget * 2 // 3
            tail_chars = budget - head_chars
            notice = f"\n\n... truncated ({total_chars} chars total) ...\n\n"
            return text[:head_chars] + notice + text[-tail_chars:]

        head_lines = lines[:head_count]
        tail_lines = lines[-tail_count:]
        middle_lines = lines[head_count:total_lines - tail_count]

        # Find error/warning lines in the middle
        error_lines = [line for line in middle_lines if _ERROR_PATTERNS.search(line)]

        head_text = "".join(head_lines)
        tail_text = "".join(tail_lines)

        # Build result, trimming if over budget
        omitted = total_lines - head_count - tail_count - len(error_lines)
        notice = f"\n\n... {omitted} lines omitted ({total_chars} chars total) ...\n\n"

        result = head_text + notice
        if error_lines:
            result += "".join(error_lines) + "\n"
        result += tail_text

        # If still over budget, trim the error lines section first, then fall back
        if len(result) > MAX_TOOL_OUTPUT:
            # Drop error lines and use char-based head/tail
            available = budget - len(notice)
            head_share = int(available * 0.40)
            tail_share = available - head_share
            result = text[:head_share] + notice + text[-tail_share:]

        return result

    else:
        # Character-based truncation: first 40%, last 20%
        head_chars = int(budget * 0.40)
        tail_chars = int(budget * 0.20)
        # Give remaining budget to head
        head_chars += budget - head_chars - tail_chars

        notice = f"\n\n... truncated ({total_chars} chars total) ...\n\n"
        return text[:head_chars] + notice + text[-tail_chars:]


def _check_bash_safety(command: str) -> str | None:
    """Return an error message if the command is unsafe, else None.

    Uses the hardened check_command_safety() from security.py which combines
    regex patterns, command normalization, shlex parsing, and obfuscation detection.
    """
    return check_command_safety(command)


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities to produce readable text."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Convert common block elements to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BINARY_MIME_PREFIXES = (
    "application/octet-stream", "application/zip", "application/gzip",
    "application/x-tar", "application/pdf", "application/x-executable",
    "application/x-sharedlib", "application/java-archive",
    "application/vnd.", "audio/", "video/", "font/",
)


_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


def _save_binary_download(content: bytes, url: str, content_type: str) -> str:
    """Save binary content to workspace/downloads/ and return a description."""
    if len(content) > _MAX_DOWNLOAD_SIZE:
        return f"[error: file too large ({len(content):,} bytes, max {_MAX_DOWNLOAD_SIZE // 1024 // 1024}MB)]"
    ct_display = content_type.split(";")[0].strip() if content_type else "unknown"
    if _workspace_path:
        downloads_dir = Path(_workspace_path) / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        url_path = urlparse(url).path
        filename = Path(url_path).name if Path(url_path).name else f"download_{uuid.uuid4().hex[:8]}"
        filepath = downloads_dir / filename
        filepath.write_bytes(content)
        hint = "Use read_pdf to read it." if filepath.suffix.lower() == ".pdf" else (
            "Use read_file, bash, or other tools to inspect it."
        )
        return (
            f"[Binary file downloaded: downloads/{filename} "
            f"({len(content):,} bytes, type: {ct_display})]\n"
            f"{hint}"
        )
    return f"[Binary content ({len(content):,} bytes, type: {ct_display}). Cannot display as text.]"


# ── Binary detection ─────────────────────────────────────────────────

_BINARY_EXTENSIONS = frozenset({
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Java / compiled / object
    ".jar", ".war", ".class", ".o", ".so", ".dylib", ".dll", ".exe",
    ".pyc", ".pyo", ".wasm",
    # Databases
    ".db", ".sqlite", ".sqlite3",
    # Office / documents (zip-based)
    ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub",
    # Packages
    ".apk", ".ipa", ".deb", ".rpm",
    # Raw binary
    ".bin", ".dat",
    # Media (non-image — images handled by is_image())
    ".ico", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".ogg",
    ".mkv", ".wmv", ".aac", ".m4a",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
})


def _is_binary(path: Path) -> bool:
    """Detect whether a file is binary.

    Checks extension against a known set first, then reads the first 8KB
    and looks for null bytes or a high ratio of non-text bytes.  This
    catches domain-specific binary formats (e.g. .mxl, .mdb) that aren't
    in the extension list.
    """
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    # Count non-text bytes (excluding tab=0x09, newline=0x0A, CR=0x0D)
    non_text = sum(1 for b in chunk if b < 0x09 or (0x0E <= b <= 0x1F))
    return (non_text / len(chunk)) > 0.10


def _binary_file_message(path: Path) -> str:
    """Build an informative message when a binary file is detected."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    ext = path.suffix.lower() or "(no extension)"
    return (
        f"[Binary file: {path.name} ({size:,} bytes, extension: {ext})]\n"
        "This is a binary file and cannot be displayed as text.\n"
        f"To inspect it, use bash: file {path}, xxd {path} | head, or strings {path}\n"
        "To work with binary formats, install tools you need with: "
        "apt-get install -y <package> or pip install <package>"
    )


# ── Tool executors ────────────────────────────────────────────────────

async def _exec_bash(args: dict) -> str:
    command = args["command"]
    # Safety check
    blocked = _check_bash_safety(command)
    if blocked:
        return blocked
    timeout = min(args.get("timeout", 60), 300)
    try:
        if _shell is not None:
            stdout_str, stderr_str, code = await _shell.execute(command, timeout=timeout)
            if code == -1 and not stdout_str and not stderr_str:
                return f"[timed out after {timeout}s]"
            # Check for binary output (null bytes in raw stdout)
            if "\x00" in stdout_str and len(stdout_str) > 64:
                out = (
                    f"[binary output detected ({len(stdout_str):,} chars) — not displayed to avoid chat corruption]\n"
                    "Hint: redirect binary output to a file instead, or use tools like xxd/hexdump."
                )
            else:
                out = stdout_str
                # Secondary check: excessive replacement chars indicate binary data
                if len(out) > 200:
                    replacement_count = out.count("\ufffd")
                    if replacement_count > 0 and replacement_count / len(out) > 0.05:
                        out = (
                            out[:200]
                            + f"\n\n[truncated: output contains binary data "
                            f"({replacement_count} invalid bytes in {len(out)} chars)]"
                        )
            err = stderr_str
        else:
            # Fallback: one-shot subprocess (shell not initialized)
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            # Check for binary output before decoding
            if b"\x00" in stdout and len(stdout) > 64:
                out = (
                    f"[binary output detected ({len(stdout):,} bytes) — not displayed to avoid chat corruption]\n"
                    "Hint: redirect binary output to a file instead, or use tools like xxd/hexdump."
                )
            else:
                out = stdout.decode("utf-8", errors="replace")
                # Secondary check: excessive replacement chars indicate binary data
                if len(out) > 200:
                    replacement_count = out.count("\ufffd")
                    if replacement_count > 0 and replacement_count / len(out) > 0.05:
                        out = (
                            out[:200]
                            + f"\n\n[truncated: output contains binary data "
                            f"({replacement_count} invalid bytes in {len(out)} chars)]"
                        )
            err = stderr.decode("utf-8", errors="replace")
            code = proc.returncode or 0
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit code: {code}]")
        return _truncate("\n".join(parts), source="bash")
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
            await proc.wait()
        except (ProcessLookupError, NameError):
            pass
        return f"[timed out after {timeout}s]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_read_file(args: dict, *, image_callback=None) -> str:
    path = args["path"]
    offset = args.get("offset", 1)
    limit = args.get("limit")
    try:
        if _workspace_path:
            resolved = validate_workspace_path(path, _workspace_path, must_exist=True)
        else:
            resolved = Path(path)
        # Image detection
        if is_image(str(resolved)):
            blocks = build_image_content_blocks(
                str(resolved), annotation=f"[Image: {path}]"
            )
            if image_callback:
                image_callback(blocks)
            return f"[Read image: {path}]"
        # PDF: redirect to read_pdf
        if resolved.suffix.lower() == ".pdf":
            return f"[This is a PDF file. Use the read_pdf tool to read it: read_pdf(path=\"{path}\")]"
        # Binary detection — after image/PDF checks
        if _is_binary(resolved):
            return _binary_file_message(resolved)
        with open(resolved, "r", errors="replace") as f:
            content = f.read()
        if context := args.get("_context"):
            if isinstance(context, ToolExecutionContext):
                context.read_hashes[str(resolved.resolve())] = hashlib.sha256(
                    content.encode()
                ).hexdigest()
        lines = content.splitlines(keepends=True)
        start = max(0, offset - 1)
        end = start + limit if limit else len(lines)
        numbered = [f"{i + start + 1}\t{line}" for i, line in enumerate(lines[start:end])]
        return _truncate("".join(numbered), source="read_file") if numbered else "(empty file)"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except FileNotFoundError:
        return f"[error: file not found: {path}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_read_many_files(args: dict) -> str:
    paths = args.get("paths")
    if not isinstance(paths, list) or not paths:
        return "[error: missing required 'paths' array]"
    if len(paths) > 20:
        return "[error: read_many_files accepts at most 20 paths]"
    offset = args.get("offset", 1)
    limit = args.get("limit")
    context = args.get("_context")
    parts = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            parts.append("== <invalid path> ==\n[error: path must be a non-empty string]")
            continue
        read_args = {"path": raw_path, "offset": offset}
        if limit is not None:
            read_args["limit"] = limit
        if isinstance(context, ToolExecutionContext):
            read_args["_context"] = context
        content = await _exec_read_file(read_args)
        parts.append(f"== {raw_path} ==\n{content}")
    return _truncate("\n\n".join(parts), source="read_many_files")


def _parse_page_ranges(spec: str, max_page: int) -> list[int]:
    """Parse a page spec like '1', '1-3', '2,5,8' into a sorted list of 0-based indices."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(max_page, int(end))
            pages.update(range(start - 1, end))
        else:
            p = int(part)
            if 1 <= p <= max_page:
                pages.add(p - 1)
    return sorted(pages)


MAX_PDF_PAGES_IMAGE = 3
MAX_PDF_PAGES_TEXT = 20


async def _exec_read_pdf(args: dict, *, image_callback=None) -> str:
    path = args["path"]
    page_spec = args.get("pages", "1")
    mode = args.get("mode", "text")
    try:
        if _workspace_path:
            resolved = validate_workspace_path(path, _workspace_path, must_exist=True)
        else:
            resolved = Path(path)
    except PathSecurityError as e:
        return f"[error: {e}]"
    except FileNotFoundError:
        return f"[error: file not found: {path}]"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[error: PDF support not available (pymupdf not installed)]"

    try:
        doc = fitz.open(str(resolved))
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return "[error: PDF has no pages]"

        max_pages = MAX_PDF_PAGES_IMAGE if mode == "image" else MAX_PDF_PAGES_TEXT
        page_indices = _parse_page_ranges(page_spec, total_pages)
        if not page_indices:
            doc.close()
            return f"[error: no valid pages in '{page_spec}' (PDF has {total_pages} pages)]"
        if len(page_indices) > max_pages:
            page_indices = page_indices[:max_pages]

        if mode == "image":
            blocks: list[dict] = []
            for idx in page_indices:
                page = doc[idx]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                from baal_agent.image_utils import resize_image_bytes
                resized = resize_image_bytes(img_bytes, max_dim=1024)
                mime = "image/jpeg" if resized is not img_bytes else "image/png"
                b64 = base64.b64encode(resized).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"
                blocks.append({"type": "text", "text": f"[PDF page {idx + 1}/{total_pages}: {path}]"})
                blocks.append({"type": "image_url", "image_url": {"url": data_uri}})

            doc.close()

            if image_callback:
                image_callback(blocks)

            rendered = [str(i + 1) for i in page_indices]
            return f"[Read PDF: {path} — page(s) {', '.join(rendered)} of {total_pages}]"
        else:
            # Text extraction mode
            parts = []
            for idx in page_indices:
                page = doc[idx]
                text = page.get_text()
                header = f"── Page {idx + 1}/{total_pages} ──"
                parts.append(f"{header}\n{text.strip()}" if text.strip() else f"{header}\n(no text content)")

            doc.close()
            result = f"[PDF: {path} — {total_pages} pages total]\n\n" + "\n\n".join(parts)
            return _truncate(result, source="read_pdf")
    except Exception as e:
        return f"[error reading PDF: {e}]"


async def _exec_write_file(args: dict) -> str:
    path = args.get("path")
    content = args.get("content")
    if not path:
        return "[error: missing required 'path' parameter]"
    if content is None:
        return "[error: missing required 'content' parameter]"
    try:
        if _workspace_path:
            resolved = validate_workspace_path(path, _workspace_path)
        else:
            resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_edit_file(args: dict) -> str:
    path = args.get("path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    replace_all = bool(args.get("replace_all", False))
    expected_hash = args.get("expected_hash")
    if not path:
        return "[error: missing required 'path' parameter]"
    if old_string is None:
        return "[error: missing required 'old_string' parameter]"
    if new_string is None:
        return "[error: missing required 'new_string' parameter]"
    try:
        if _workspace_path:
            resolved = validate_workspace_path(path, _workspace_path, must_exist=True)
        else:
            resolved = Path(path)
        if _is_binary(resolved):
            return f"[error: {path} is a binary file and cannot be edited as text]"
        with open(resolved, "r") as f:
            content = f.read()
        actual_hash = hashlib.sha256(content.encode()).hexdigest()
        if expected_hash:
            if actual_hash != expected_hash:
                return (
                    f"[error: stale edit refused for {path}; expected_hash "
                    f"does not match current file hash]"
                )
        else:
            context = args.get("_context")
            read_hash = (
                context.read_hashes.get(str(resolved.resolve()))
                if isinstance(context, ToolExecutionContext)
                else None
            )
            if not read_hash:
                return (
                    f"[error: edit refused for {path}; read the file in this "
                    "turn first or pass expected_hash]"
                )
            if read_hash != actual_hash:
                return (
                    f"[error: stale edit refused for {path}; file changed "
                    "since it was read this turn]"
                )
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"[error: old_string not found in {path}]"
        if occurrences > 1 and not replace_all:
            return (
                f"[error: old_string appears {occurrences} times in {path}; "
                "refusing ambiguous edit. Set replace_all=true to replace all occurrences.]"
            )
        content = content.replace(
            old_string, new_string, occurrences if replace_all else 1
        )
        with open(resolved, "w") as f:
            f.write(content)
        context = args.get("_context")
        if isinstance(context, ToolExecutionContext):
            context.read_hashes[str(resolved.resolve())] = hashlib.sha256(
                content.encode()
            ).hexdigest()
        if replace_all:
            return f"Edited {path} ({occurrences} replacements)"
        return f"Edited {path}"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except FileNotFoundError:
        return f"[error: file not found: {path}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_multi_edit(args: dict) -> str:
    edits = args.get("edits")
    if not isinstance(edits, list) or not edits:
        return "[error: missing required 'edits' array]"
    if len(edits) > 50:
        return "[error: multi_edit accepts at most 50 edits]"
    context = args.get("_context")
    results = []
    for idx, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            return f"[error: edit {idx} must be an object]"
        edit_args = dict(edit)
        if isinstance(context, ToolExecutionContext):
            edit_args["_context"] = context
        result = await _exec_edit_file(edit_args)
        results.append(f"{idx}. {result}")
        if _is_error_result(result):
            results.append("Stopped after first failed edit.")
            break
    return "\n".join(results)


async def _run_exec(
    argv: list[str],
    *,
    source: str,
    timeout: int = 60,
    cwd: Path | None = None,
) -> str:
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    run_cwd = cwd or workspace
    timeout = min(max(int(timeout), 1), 600)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(run_cwd),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out:
            parts.append(out.rstrip())
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        parts.append(f"[exit code: {proc.returncode or 0}]")
        return _truncate("\n".join(parts), source=source)
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
            await proc.wait()
        except (NameError, ProcessLookupError):
            pass
        return f"[timed out after {timeout}s]"
    except FileNotFoundError:
        return f"[error: command not found: {argv[0]}]"
    except Exception as e:
        return f"[error: {e}]"


async def _run_shell_command_tool(args: dict, *, source: str) -> str:
    command = args.get("command")
    if not command:
        return "[error: missing required 'command' parameter]"
    blocked = _check_bash_safety(command)
    if blocked:
        return blocked
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    timeout = min(max(int(args.get("timeout", 120)), 1), 600)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out:
            parts.append(out.rstrip())
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        parts.append(f"[exit code: {proc.returncode or 0}]")
        return _truncate("\n".join(parts), source=source)
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
            await proc.wait()
        except (NameError, ProcessLookupError):
            pass
        return f"[timed out after {timeout}s]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_apply_patch(args: dict) -> str:
    patch = args.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        return "[error: missing required 'patch' parameter]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    git = shutil.which("git")
    if not git:
        return "[error: git is not installed]"
    try:
        _validate_patch_paths(patch, workspace)
    except PathSecurityError as e:
        return f"[error: {e}]"
    except ValueError as e:
        return f"[error: {e}]"

    check_proc = await asyncio.create_subprocess_exec(
        git,
        "apply",
        "--check",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )
    check_stdout, check_stderr = await asyncio.wait_for(
        check_proc.communicate(patch.encode()), timeout=30
    )
    if check_proc.returncode != 0:
        detail = (check_stderr or check_stdout).decode("utf-8", errors="replace").strip()
        return f"[error: patch check failed: {detail}]"

    paths_ok, paths_err = await _verify_patch_numstat(git, workspace, patch)
    if not paths_ok:
        return paths_err

    apply_proc = await asyncio.create_subprocess_exec(
        git,
        "apply",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )
    stdout, stderr = await asyncio.wait_for(apply_proc.communicate(patch.encode()), timeout=30)
    if apply_proc.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
        return f"[error: patch apply failed: {detail}]"
    return "Patch applied"


async def _verify_patch_numstat(
    git: str, workspace: Path, patch: str
) -> tuple[bool, str]:
    """Re-validate the patch via git's own filename parser.

    `_validate_patch_paths` is a best-effort header parse and is fragile
    against quoted filenames with embedded escapes; git's --numstat output
    is the canonical source of truth for which paths a patch would touch.
    """
    proc = await asyncio.create_subprocess_exec(
        git,
        "apply",
        "--numstat",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(patch.encode()), timeout=30)
    try:
        for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
            parts = raw_line.split("\t")
            if len(parts) < 3:
                continue
            touched = parts[2].strip()
            if not touched or touched.startswith("/"):
                return False, f"[error: patch touches non-workspace path: {touched!r}]"
            candidates = [touched]
            if " => " in touched:
                left, _, right = touched.partition(" => ")
                candidates = [left.strip("{}"), right.strip("{}")]
            for candidate in candidates:
                if not candidate or candidate == "/dev/null":
                    continue
                validate_workspace_path(candidate, workspace, must_exist=False)
    except PathSecurityError as e:
        return False, f"[error: patch touches path outside workspace: {e}]"
    return True, ""


def _validate_patch_paths(patch: str, workspace: Path) -> None:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.update(parts[2:4])
        elif line.startswith("--- ") or line.startswith("+++ "):
            raw = line[4:].split("\t", 1)[0].strip()
            paths.add(raw)

    for raw_path in paths:
        if raw_path == "/dev/null":
            continue
        path = raw_path
        if path.startswith(("a/", "b/")):
            path = path[2:]
        if not path or path.startswith("/") or Path(path).is_absolute():
            raise ValueError(f"patch path escapes workspace: {raw_path}")
        validate_workspace_path(path, workspace, must_exist=False)


def _validate_git_path_arg(path: str | None, workspace: Path) -> str | None:
    if not path:
        return None
    resolved = validate_workspace_path(path, workspace, must_exist=False)
    return str(resolved.relative_to(workspace))


def _skill_load_metadata(name: str, arguments: dict) -> list[dict]:
    if name not in {"read_file", "read_many_files"} or not _workspace_path:
        return []
    workspace = Path(_workspace_path).resolve()
    raw_paths = (
        arguments.get("paths", [])
        if name == "read_many_files"
        else [arguments.get("path")]
    )
    loaded = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        try:
            resolved = validate_workspace_path(raw_path, workspace, must_exist=True)
            rel = resolved.relative_to(workspace)
        except (PathSecurityError, FileNotFoundError, ValueError, OSError):
            continue
        parts = rel.parts
        if len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md":
            loaded.append({"id": parts[1], "path": str(rel)})
    return loaded


async def _exec_git_status(args: dict) -> str:
    git = shutil.which("git")
    if not git:
        return "[error: git is not installed]"
    return await _run_exec([git, "status", "--short", "--branch"], source="git_status")


async def _exec_git_diff(args: dict) -> str:
    git = shutil.which("git")
    if not git:
        return "[error: git is not installed]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    cmd = [git, "diff"]
    if args.get("staged"):
        cmd.append("--cached")
    try:
        path = _validate_git_path_arg(args.get("path"), workspace)
        if path:
            cmd.extend(["--", path])
        return await _run_exec(cmd, source="git_diff")
    except PathSecurityError as e:
        return f"[error: {e}]"


async def _exec_git_show(args: dict) -> str:
    git = shutil.which("git")
    if not git:
        return "[error: git is not installed]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    rev = args.get("rev", "HEAD")
    if not isinstance(rev, str) or not rev:
        return "[error: rev must be a non-empty string]"
    cmd = [git, "show", "--stat", "--patch", "--", rev]
    try:
        path = _validate_git_path_arg(args.get("path"), workspace)
        if path:
            cmd = [git, "show", "--stat", "--patch", rev, "--", path]
        else:
            cmd = [git, "show", "--stat", "--patch", rev]
        return await _run_exec(cmd, source="git_show")
    except PathSecurityError as e:
        return f"[error: {e}]"


async def _exec_git_blame(args: dict) -> str:
    path = args.get("path")
    if not path:
        return "[error: missing required 'path' parameter]"
    git = shutil.which("git")
    if not git:
        return "[error: git is not installed]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    workspace = Path(_workspace_path).resolve()
    try:
        rel_path = _validate_git_path_arg(path, workspace)
        cmd = [git, "blame"]
        start = args.get("start")
        end = args.get("end")
        if start is not None:
            start_i = max(int(start), 1)
            if end is not None:
                end_i = max(int(end), start_i)
                cmd.extend(["-L", f"{start_i},{end_i}"])
            else:
                cmd.extend(["-L", f"{start_i},+40"])
        cmd.extend(["--", rel_path or path])
        return await _run_exec(cmd, source="git_blame")
    except PathSecurityError as e:
        return f"[error: {e}]"
    except ValueError:
        return "[error: start/end must be integers]"


async def _exec_run_tests(args: dict) -> str:
    return await _run_shell_command_tool(args, source="run_tests")


async def _exec_run_lint(args: dict) -> str:
    return await _run_shell_command_tool(args, source="run_lint")


async def _exec_run_typecheck(args: dict) -> str:
    return await _run_shell_command_tool(args, source="run_typecheck")


async def _exec_run_format(args: dict) -> str:
    return await _run_shell_command_tool(args, source="run_format")


async def _exec_list_dir(args: dict) -> str:
    path = args.get("path", ".")
    try:
        if _workspace_path:
            resolved = validate_workspace_path(path, _workspace_path, must_exist=True)
        else:
            resolved = Path(path)
        if not resolved.is_dir():
            return f"[error: not a directory: {path}]"
        entries = sorted(resolved.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries:
            prefix = "[dir]" if entry.is_dir() else "[file]"
            lines.append(f"{prefix}  {entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except PermissionError:
        return f"[error: permission denied: {path}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_glob(args: dict) -> str:
    pattern = args.get("pattern")
    if not pattern:
        return "[error: missing required 'pattern' parameter]"
    limit = min(max(int(args.get("limit", 100)), 1), 500)
    path = args.get("path", ".")
    if not _workspace_path:
        return "[error: workspace not configured]"
    try:
        workspace = Path(_workspace_path).resolve()
        root = validate_workspace_path(path, workspace, must_exist=True)
        if not root.is_dir():
            return f"[error: not a directory: {path}]"
        matches = []
        for item in root.glob(pattern):
            try:
                resolved = item.resolve()
                resolved.relative_to(workspace)
            except (OSError, ValueError):
                continue
            matches.append(resolved)
        matches = sorted(
            matches,
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not matches:
            return "(no matches)"
        lines = []
        for match in matches[:limit]:
            rel = match.relative_to(workspace)
            suffix = "/" if match.is_dir() else ""
            lines.append(f"{rel}{suffix}")
        if len(matches) > limit:
            lines.append(f"... {len(matches) - limit} more matches omitted")
        return "\n".join(lines)
    except PathSecurityError as e:
        return f"[error: {e}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_grep(args: dict) -> str:
    pattern = args.get("pattern")
    if not pattern:
        return "[error: missing required 'pattern' parameter]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    rg = shutil.which("rg")
    if not rg:
        return "[error: ripgrep (rg) is not installed]"
    output_mode = args.get("output_mode", "content")
    if output_mode not in {"content", "files_with_matches", "count"}:
        return "[error: output_mode must be one of content, files_with_matches, count]"
    limit = min(max(int(args.get("limit", 200)), 1), 1000)
    path = args.get("path", ".")
    try:
        workspace = Path(_workspace_path).resolve()
        target = validate_workspace_path(path, workspace, must_exist=True)
        cmd = [rg, "--color", "never"]
        if output_mode == "content":
            cmd.extend(["--line-number", "--no-heading"])
        elif output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count-matches")
        if args.get("case_sensitive") is False:
            cmd.append("--ignore-case")
        type_filter = args.get("type_filter")
        if type_filter:
            cmd.extend(["--type", str(type_filter)])
        cmd.extend(["--", str(pattern), str(target)])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode == 1 and not out:
            return "(no matches)"
        if proc.returncode not in (0, 1):
            return f"[error: rg exited {proc.returncode}: {err or out}]"
        lines = out.splitlines()
        if len(lines) > limit:
            out = "\n".join(lines[:limit])
            out += f"\n... {len(lines) - limit} more lines omitted"
        return _truncate(out, source="grep") if out.strip() else "(no matches)"
    except asyncio.TimeoutError:
        return "[timed out after 30s]"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_web_fetch(args: dict, *, image_callback=None) -> str:
    url = args["url"]
    if not re.match(r"^https?://", url):
        return "[error: URL must start with http:// or https://]"
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, max_redirects=5) as client:
            resp = await client.get(url, headers={"User-Agent": "BaalAgent/1.0"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            # Image detection by URL extension or content type
            if is_image(urlparse(url).path) or content_type.startswith("image/"):
                data_uri = encode_bytes_to_data_uri(
                    resp.content, mime=content_type.split(";")[0] or "image/jpeg"
                )
                blocks: list[dict] = [
                    {"type": "text", "text": f"[Image: {url}]"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
                if image_callback:
                    image_callback(blocks)
                return f"[Fetched image: {url}]"
            # Binary content detection by MIME type or content sniff
            if any(content_type.startswith(p) for p in _BINARY_MIME_PREFIXES) or (
                b"\x00" in resp.content[:8192]
            ):
                return _save_binary_download(resp.content, url, content_type)
            text = resp.text
            if "json" in content_type:
                try:
                    parsed = json.loads(text)
                    text = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass
            elif "html" in content_type:
                text = _strip_html(text)
            if len(text) > MAX_WEB_CONTENT:
                text = text[:MAX_WEB_CONTENT] + f"\n\n... truncated ({len(resp.text)} chars total)"
            return text if text.strip() else "(empty response)"
    except httpx.HTTPStatusError as e:
        return f"[error: HTTP {e.response.status_code}]"
    except Exception as e:
        return f"[error: {e}]"


# ── Browser tool (headless Chromium via Playwright) ───────────────────
# A single headless browser/context is launched lazily and reused for the
# whole process so navigation state (cookies, current page) persists across
# tool calls. A lock serializes concurrent calls; the dispatcher also treats
# `browser` as mutating so calls run sequentially.

BROWSER_NAV_TIMEOUT_MS = 30_000
_browser_playwright = None  # the started Playwright context manager
_browser = None  # launched Chromium Browser
_browser_page = None  # the single active Page
_browser_lock = asyncio.Lock()

# Only these schemes may be navigated. An explicit allowlist (rather than a
# regex prefix) keeps file:/data:/chrome:/view-source: and friends out.
_ALLOWED_BROWSER_SCHEMES = {"http", "https"}


def _ip_is_blocked(ip_str: str) -> bool:
    """True if ``ip_str`` is a non-public SSRF target (private, loopback,
    link-local, reserved, multicast or unspecified). Unparseable → blocked.
    IPv4-mapped IPv6 is unwrapped so ``::ffff:10.0.0.1`` is caught."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _validate_public_url(url: str) -> str | None:
    """SSRF guard for the browser tool. Returns an error reason when ``url`` is
    not a safe, public http(s) target, else ``None``.

    Rejects non-http(s) schemes and any host that resolves (via getaddrinfo) to
    a private/loopback/link-local/reserved/multicast address — blocking cloud
    metadata (169.254.169.254), localhost and RFC1918 pivots. Re-run on every
    navigation by ``_ssrf_route_guard`` so redirects can't slip past."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "invalid URL"
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_BROWSER_SCHEMES:
        return f"scheme '{scheme or '?'}' not allowed — only http/https"
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        return "URL has no host"
    port = parsed.port or (443 if scheme == "https" else 80)
    loop = asyncio.get_event_loop()
    try:
        infos = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP),
        )
    except OSError as e:
        return f"could not resolve host '{hostname}': {e}"
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return f"could not resolve host '{hostname}'"
    for ip_str in addrs:
        if _ip_is_blocked(ip_str):
            return f"blocked: '{hostname}' resolves to non-public address {ip_str}"
    return None


async def _ssrf_route_guard(route) -> None:
    """Context route handler: re-validate every top-level navigation (initial
    load, server redirect, history back/forward) against the SSRF policy and
    abort ones aimed at non-public hosts. Subresources pass through — their
    bodies are never returned to the model."""
    request = route.request
    is_nav = False
    try:
        is_nav = request.is_navigation_request()
        if is_nav and await _validate_public_url(request.url) is not None:
            await route.abort("blockedbyclient")
            return
        await route.continue_()
    except Exception:
        # Fail closed for navigations, fail open for subresources.
        try:
            await (route.abort("failed") if is_nav else route.continue_())
        except Exception:
            pass


async def _get_browser_page():
    """Lazily launch (and reuse) a headless Chromium page.

    Raises RuntimeError on launch failure so the caller can translate it into
    an ``[error: ...]`` result string.
    """
    global _browser_playwright, _browser, _browser_page
    if _browser_page is not None:
        return _browser_page
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise RuntimeError(f"playwright is not installed: {e}") from e
    if _browser_playwright is None:
        _browser_playwright = await async_playwright().start()
    if _browser is None:
        _browser = await _browser_playwright.chromium.launch(headless=True)
    context = await _browser.new_context()
    # Re-validate every navigation (incl. redirects/back) against the SSRF
    # policy, not just the initial goto URL.
    await context.route("**/*", _ssrf_route_guard)
    _browser_page = await context.new_page()
    _browser_page.set_default_timeout(BROWSER_NAV_TIMEOUT_MS)
    return _browser_page


async def shutdown_browser() -> None:
    """Close the shared browser. Safe to call even if never launched."""
    global _browser_playwright, _browser, _browser_page
    try:
        if _browser is not None:
            await _browser.close()
    except Exception:
        pass
    try:
        if _browser_playwright is not None:
            await _browser_playwright.stop()
    except Exception:
        pass
    _browser_playwright = None
    _browser = None
    _browser_page = None


async def _exec_browser(args: dict, *, image_callback=None) -> str:
    action = str(args.get("action", "")).strip()
    if not action:
        return "[error: missing required 'action' parameter]"
    if action not in {"goto", "extract", "click", "type", "screenshot", "back"}:
        return f"[error: unknown browser action '{action}']"

    if action == "goto":
        url = args.get("url")
        if not url:
            return "[error: 'goto' requires a 'url' parameter]"
        nav_error = await _validate_public_url(url)
        if nav_error is not None:
            return f"[error: {nav_error}]"

    async with _browser_lock:
        try:
            page = await _get_browser_page()
        except RuntimeError as e:
            return f"[error: {e}]"
        try:
            if action == "goto":
                await page.goto(args["url"], timeout=BROWSER_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                return f"Navigated to {page.url} — \"{await page.title()}\""

            if action == "back":
                await page.go_back(timeout=BROWSER_NAV_TIMEOUT_MS)
                return f"Went back to {page.url} — \"{await page.title()}\""

            if action == "click":
                selector = args.get("selector")
                if not selector:
                    return "[error: 'click' requires a 'selector' parameter]"
                await page.click(selector, timeout=BROWSER_NAV_TIMEOUT_MS)
                return f"Clicked {selector!r}. Now at {page.url}"

            if action == "type":
                selector = args.get("selector")
                text = args.get("text")
                if not selector:
                    return "[error: 'type' requires a 'selector' parameter]"
                if text is None:
                    return "[error: 'type' requires a 'text' parameter]"
                await page.fill(selector, str(text), timeout=BROWSER_NAV_TIMEOUT_MS)
                return f"Typed into {selector!r}."

            if action == "extract":
                content = await page.content()
                text = _strip_html(content)
                if len(text) > MAX_WEB_CONTENT:
                    text = text[:MAX_WEB_CONTENT] + f"\n\n... truncated ({len(content)} chars total)"
                return text if text.strip() else "(empty page)"

            if action == "screenshot":
                png = await page.screenshot(type="png")
                data_uri = encode_bytes_to_data_uri(png, mime="image/png")
                blocks: list[dict] = [
                    {"type": "text", "text": f"[Screenshot: {page.url}]"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
                if image_callback:
                    image_callback(blocks)
                return f"[Captured screenshot of {page.url}]"
        except Exception as e:
            return f"[error: {e}]"

    return f"[error: unknown browser action '{action}']"


async def _exec_search_history(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "[error: missing required 'query' parameter]"
    if _db is None:
        return "[error: conversation search not available]"
    chat_id = args.get("chat_id")
    limit = min(args.get("limit", 20), 50)
    summarize = args.get("summarize", False)
    try:
        results = await _db.search_history(query, chat_id=chat_id, limit=limit)
    except Exception as e:
        return f"[error: search failed: {e}]"
    if not results:
        return "(no matching messages found)"
    lines = []
    for r in results:
        lines.append(f"[{r['created_at']}] ({r['role']}, chat: {r['chat_id']}):")
        lines.append(r["snippet"])
        lines.append("")
    raw_output = "\n".join(lines)

    if summarize and _inference and _model:
        try:
            summary_prompt = (
                f'Based on the following search results from conversation history, '
                f'provide a coherent summary that answers the query "{query}":\n\n'
                f'{raw_output}\n\n'
                f'Synthesize the key information concisely.'
            )
            response = await _inference.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                model=_model,
            )
            summary = response.content
            if summary:
                return summary
        except Exception as e:
            # Fall back to raw results on summarization failure
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Search summarization failed: {e}")

    return _truncate(raw_output, source="search_history")


def _context_chat_id(args: dict) -> str | None:
    """Pull the conversation's chat_id off the injected ToolExecutionContext.

    Returns None when running unscoped (legacy paths, in-process eval, etc.).
    """
    ctx = args.get("_context")
    if isinstance(ctx, ToolExecutionContext) and ctx.chat_id:
        return ctx.chat_id
    return None


async def _exec_remember_fact(args: dict) -> str:
    if _db is None:
        return "[error: memory database is not configured]"
    kind = str(args.get("kind", "")).strip()
    content = str(args.get("content", "")).strip()
    source = str(args.get("source", "agent") or "agent").strip()
    metadata = args.get("metadata")
    if not kind:
        return "[error: missing required 'kind' parameter]"
    if not content:
        return "[error: missing required 'content' parameter]"
    if len(content) > 2_000:
        return "[error: memory content must be 2000 characters or less]"
    if metadata is not None and not isinstance(metadata, dict):
        return "[error: metadata must be an object when provided]"
    chat_id = _context_chat_id(args)
    record_id = await _db.add_memory_record(
        kind=kind,
        content=content,
        source=source,
        metadata=metadata,
        chat_id=chat_id,
    )
    return f"Stored memory record {record_id} ({kind})"


async def _exec_search_memory(args: dict) -> str:
    if _db is None:
        return "[error: memory database is not configured]"
    query = str(args.get("query", "") or "").strip()
    kind = args.get("kind")
    if kind is not None:
        kind = str(kind).strip() or None
    limit = min(max(int(args.get("limit", 20)), 1), 100)
    records = await _db.search_memory_records(
        query,
        kind=kind,
        limit=limit,
        include_archived=bool(args.get("include_archived", False)),
        chat_id=_context_chat_id(args),
    )
    if not records:
        return "(no memory records)"
    lines = []
    for record in records:
        lines.append(
            f"[{record['id']}] {record['kind']} ({record['source']}): "
            f"{record['content']}"
        )
    return _truncate("\n".join(lines), source="search_memory")


async def _exec_web_search(args: dict) -> str:
    query = args["query"]
    count = min(args.get("count", 5), 10)
    api_key = os.environ.get("LIBERTAI_API_KEY", "")
    if not api_key:
        return "[error: LIBERTAI_API_KEY not configured]"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://search.libertai.io/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "engines": ["google", "bing", "duckduckgo"],
                    "max_results": count,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            meta = data.get("meta", {})
            failed = meta.get("engines_failed", [])
            used = meta.get("engines_used", [])
            requested = ["google", "bing", "duckduckgo"]

            # All engines failed — no usable results
            if len(failed) >= len(requested) or (not results and failed):
                return (
                    "[error: all search engines failed — search service is "
                    "currently unavailable. Use web_fetch to search specific "
                    "sites directly instead, e.g.:\n"
                    "- https://arxiv.org/search/?query=YOUR+QUERY\n"
                    "- https://scholar.google.com/scholar?q=YOUR+QUERY\n"
                    "- https://en.wikipedia.org/wiki/TOPIC]"
                )

            if not results:
                return "(no results found)"

            lines = []

            # Prominent warning when most engines are down
            if len(failed) >= 2:
                lines.append(
                    f"[warning: {len(failed)}/{len(requested)} search engines "
                    f"failed ({', '.join(failed)}). Only {', '.join(used)} "
                    "returned results — quality may be poor. Consider using "
                    "web_fetch on specific sites for better results.]"
                )
                lines.append("")

            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                lines.append(f"**{title}**\n{url}\n{snippet}\n")
            if failed and len(failed) < 2:
                lines.append(f"(engines failed: {', '.join(failed)})")
            return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


async def _exec_generate_image(args: dict) -> str:
    prompt = args.get("prompt")
    if not prompt:
        return "[error: missing required 'prompt' parameter]"
    api_key = os.environ.get("LIBERTAI_API_KEY", "")
    if not api_key:
        return "[error: LIBERTAI_API_KEY not configured]"
    if not _workspace_path:
        return "[error: workspace not configured]"

    # Parse and validate size
    size_str = args.get("size", "1024x1024")
    try:
        w, h = size_str.lower().split("x")
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        return f"[error: invalid size format '{size_str}', expected 'WxH' e.g. '1024x1024']"
    w = min(w, 1024)
    h = min(h, 1024)
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    size = f"{w}x{h}"

    steps = args.get("steps", 8)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.libertai.io/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "z-image-turbo",
                    "prompt": prompt,
                    "size": size,
                    "n": 1,
                    "steps": steps,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            b64_data = data["data"][0]["b64_json"]
            image_bytes = base64.b64decode(b64_data)

        # Save to workspace/images/
        images_dir = Path(_workspace_path) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        image_path = images_dir / filename
        image_path.write_bytes(image_bytes)

        rel_path = f"images/{filename}"
        return f"__SEND_FILE__:{rel_path}:{prompt}"
    except httpx.HTTPStatusError as e:
        return f"[error: HTTP {e.response.status_code} from image API]"
    except (KeyError, IndexError):
        return "[error: unexpected response format from image API]"
    except Exception as e:
        return f"[error: {e}]"


async def _exec_send_file(args: dict) -> str:
    path = args.get("path")
    caption = args.get("caption", "")
    if not path:
        return "[error: missing required 'path' parameter]"
    if not _workspace_path:
        return "[error: workspace not configured]"
    try:
        resolved = validate_workspace_path(
            path, _workspace_path, must_exist=True, reject_sensitive=True
        )
        size = resolved.stat().st_size
        if size > MAX_SEND_FILE_SIZE:
            return f"[error: file too large ({size} bytes, max {MAX_SEND_FILE_SIZE})]"
        rel = resolved.relative_to(Path(_workspace_path).resolve())
        return f"__SEND_FILE__:{rel}:{caption}"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except Exception as e:
        return f"[error: {e}]"


# ── Todo tool ────────────────────────────────────────────────────────

_TODO_VALID_STATUSES = {"pending", "in_progress", "done"}
_TODO_VALID_PRIORITIES = {"low", "medium", "high"}


def _todo_path() -> Path:
    """Return the path to the TODO.json file in the workspace."""
    if not _workspace_path:
        raise RuntimeError("workspace not configured")
    return Path(_workspace_path) / "TODO.json"


def _load_todos() -> list[dict]:
    """Load the task list from disk, returning an empty list if missing."""
    path = _todo_path()
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_todos(tasks: list[dict]) -> None:
    """Persist the task list to disk."""
    path = _todo_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)


def _next_id(tasks: list[dict]) -> int:
    """Return the next auto-incremented task ID."""
    if not tasks:
        return 1
    return max(t.get("id", 0) for t in tasks) + 1


def _format_task(t: dict) -> str:
    """Format a single task for display."""
    priority_markers = {"high": "!!!", "medium": "!!", "low": "!"}
    marker = priority_markers.get(t.get("priority", "medium"), "!!")
    status = t.get("status", "pending")
    line = f"[{t['id']}] {marker} ({status}) {t.get('title', '(untitled)')}"
    if t.get("notes"):
        line += f"\n     Notes: {t['notes']}"
    if t.get("completed_at"):
        line += f"\n     Completed: {t['completed_at']}"
    return line


async def _exec_todo(args: dict) -> str:
    action = args.get("action")
    if not action:
        return "[error: missing required 'action' parameter]"
    if not _workspace_path:
        return "[error: workspace not configured]"

    from datetime import datetime, timezone

    try:
        tasks = _load_todos()
    except Exception as e:
        return f"[error loading TODO.json: {e}]"

    if action == "add":
        title = args.get("title")
        if not title:
            return "[error: 'title' is required for add]"
        priority = args.get("priority", "medium")
        if priority not in _TODO_VALID_PRIORITIES:
            return f"[error: priority must be one of {sorted(_TODO_VALID_PRIORITIES)}]"
        task = {
            "id": _next_id(tasks),
            "title": title,
            "status": "pending",
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "notes": args.get("notes", ""),
        }
        tasks.append(task)
        _save_todos(tasks)
        return f"Added task #{task['id']}: {title}"

    elif action == "list":
        if not tasks:
            return "(no tasks)"
        # Optional status filter
        filter_status = args.get("status")
        filtered = tasks
        if filter_status:
            filtered = [t for t in tasks if t.get("status") == filter_status]
            if not filtered:
                return f"(no tasks with status '{filter_status}')"
        lines = [_format_task(t) for t in filtered]
        return "\n".join(lines)

    elif action == "update":
        task_id = args.get("id")
        if task_id is None:
            return "[error: 'id' is required for update]"
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if not task:
            return f"[error: task #{task_id} not found]"
        changed = []
        if "status" in args and args["status"] is not None:
            new_status = args["status"]
            if new_status not in _TODO_VALID_STATUSES:
                return f"[error: status must be one of {sorted(_TODO_VALID_STATUSES)}]"
            task["status"] = new_status
            changed.append(f"status={new_status}")
        if "priority" in args and args["priority"] is not None:
            new_priority = args["priority"]
            if new_priority not in _TODO_VALID_PRIORITIES:
                return f"[error: priority must be one of {sorted(_TODO_VALID_PRIORITIES)}]"
            task["priority"] = new_priority
            changed.append(f"priority={new_priority}")
        if "notes" in args and args["notes"] is not None:
            task["notes"] = args["notes"]
            changed.append("notes")
        if not changed:
            return f"[error: nothing to update for task #{task_id}]"
        _save_todos(tasks)
        return f"Updated task #{task_id}: {', '.join(changed)}"

    elif action == "complete":
        task_id = args.get("id")
        if task_id is None:
            return "[error: 'id' is required for complete]"
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if not task:
            return f"[error: task #{task_id} not found]"
        task["status"] = "done"
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save_todos(tasks)
        return f"Completed task #{task_id}: {task.get('title', '')}"

    elif action == "delete":
        task_id = args.get("id")
        if task_id is None:
            return "[error: 'id' is required for delete]"
        original_len = len(tasks)
        tasks = [t for t in tasks if t.get("id") != task_id]
        if len(tasks) == original_len:
            return f"[error: task #{task_id} not found]"
        _save_todos(tasks)
        return f"Deleted task #{task_id}"

    else:
        return f"[error: unknown action '{action}']"


# ── execute_code handler ─────────────────────────────────────────────

async def _exec_execute_code(args: dict) -> str:
    code = args.get("code")
    if not code:
        return "[error: missing required 'code' parameter]"
    if _code_executor is None:
        return "[error: code executor not available]"
    timeout = min(args.get("timeout", 120), 300)
    chat_id = _context_chat_id(args) or ""
    # Carry the parent turn's ToolPolicy into the sandbox so call_tool()
    # invocations inside the script can't bypass guardrails the outer
    # turn was running under.
    ctx = args.get("_context")
    policy = ctx.policy if isinstance(ctx, ToolExecutionContext) else None
    return await _code_executor.execute(
        code, timeout=timeout, chat_id=chat_id, policy=policy
    )


# ── Checkpoint tool ──────────────────────────────────────────────────

async def _exec_checkpoint(args: dict) -> str:
    global _checkpoint_mgr

    action = args.get("action")
    if not action:
        return "[error: missing required 'action' parameter]"
    if not _workspace_path:
        return "[error: workspace not configured]"

    # Lazy initialization on first use
    if _checkpoint_mgr is None:
        _checkpoint_mgr = CheckpointManager(_workspace_path)
    try:
        await _checkpoint_mgr.init()
    except Exception as e:
        return f"[error initializing checkpoints: {e}]"

    if not _checkpoint_mgr._initialized:
        return "[error: git is not available — checkpoints require git to be installed]"

    if action == "create":
        message = args.get("message")
        if not message:
            return "[error: 'message' is required for create]"
        try:
            result = await _checkpoint_mgr.create(message)
            if result == "no changes":
                return "No changes to checkpoint."
            if result.startswith("error:"):
                return f"[{result}]"
            return f"Checkpoint created: {result}"
        except Exception as e:
            return f"[error creating checkpoint: {e}]"

    elif action == "list":
        try:
            checkpoints = await _checkpoint_mgr.list_checkpoints()
            if not checkpoints:
                return "(no checkpoints)"
            lines = []
            for cp in checkpoints:
                lines.append(f"{cp['id']}  {cp['message']}  ({cp['timestamp']})")
            return "\n".join(lines)
        except Exception as e:
            return f"[error listing checkpoints: {e}]"

    elif action == "restore":
        cp_id = args.get("id")
        if not cp_id:
            return "[error: 'id' is required for restore]"
        try:
            result = await _checkpoint_mgr.restore(cp_id)
            if result.startswith("error:"):
                return f"[{result}]"
            return result
        except Exception as e:
            return f"[error restoring checkpoint: {e}]"

    elif action == "diff":
        cp_id = args.get("id")
        if not cp_id:
            return "[error: 'id' is required for diff]"
        try:
            result = await _checkpoint_mgr.diff(cp_id)
            if result.startswith("error:"):
                return f"[{result}]"
            return result
        except Exception as e:
            return f"[error diffing checkpoint: {e}]"

    else:
        return f"[error: unknown action '{action}']"


# ── Process management ────────────────────────────────────────────────

_MAX_PROCESSES = 10
_OUTPUT_BUFFER_SIZE = 10_240  # 10 KB
_PROCESS_RETENTION = 3600  # auto-clean completed after 1 hour


@dataclass
class ProcessInfo:
    id: str
    command: str
    process: asyncio.subprocess.Process
    status: str  # running / completed / failed
    output_buffer: str = ""
    started_at: float = _field(default_factory=_time.time)
    completed_at: float | None = None
    _reader_task: asyncio.Task | None = _field(default=None, repr=False)


_processes: dict[str, ProcessInfo] = {}


def _prune_completed_processes():
    """Remove completed processes older than _PROCESS_RETENTION."""
    cutoff = _time.time() - _PROCESS_RETENTION
    to_remove = [
        pid for pid, info in _processes.items()
        if info.status != "running" and info.completed_at and info.completed_at < cutoff
    ]
    for pid in to_remove:
        del _processes[pid]


async def _process_output_reader(info: ProcessInfo):
    """Background task that reads stdout+stderr and appends to the buffer."""
    try:
        async def _read_stream(stream):
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                # Keep only the last _OUTPUT_BUFFER_SIZE bytes
                info.output_buffer += text
                if len(info.output_buffer) > _OUTPUT_BUFFER_SIZE:
                    info.output_buffer = info.output_buffer[-_OUTPUT_BUFFER_SIZE:]

        await asyncio.gather(
            _read_stream(info.process.stdout),
            _read_stream(info.process.stderr),
        )
    except Exception:
        pass
    finally:
        # Wait for the process to finish and update status
        try:
            await info.process.wait()
        except Exception:
            pass
        code = info.process.returncode
        info.status = "completed" if code == 0 else "failed"
        info.completed_at = _time.time()
        if code is not None and code != 0:
            info.output_buffer += f"\n[exit code: {code}]"
            if len(info.output_buffer) > _OUTPUT_BUFFER_SIZE:
                info.output_buffer = info.output_buffer[-_OUTPUT_BUFFER_SIZE:]


async def _exec_process(args: dict) -> str:
    action = args.get("action")
    if not action:
        return "[error: missing required 'action' parameter]"

    _prune_completed_processes()

    if action == "start":
        command = args.get("command")
        if not command:
            return "[error: 'command' is required for start]"
        # Safety check
        blocked = _check_bash_safety(command)
        if blocked:
            return blocked
        # Enforce concurrency limit
        active = sum(1 for p in _processes.values() if p.status == "running")
        if active >= _MAX_PROCESSES:
            return f"[error: too many concurrent processes ({active}/{_MAX_PROCESSES}). Kill some first.]"
        proc_id = uuid.uuid4().hex[:8]
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_workspace_path,
            )
        except Exception as e:
            return f"[error starting process: {e}]"
        info = ProcessInfo(
            id=proc_id,
            command=command,
            process=proc,
            status="running",
        )
        info._reader_task = asyncio.create_task(_process_output_reader(info))
        _processes[proc_id] = info
        return f"Started process {proc_id} (PID {proc.pid}): {command}"

    elif action == "list":
        if not _processes:
            return "No processes."
        lines = []
        for info in _processes.values():
            pid_str = str(info.process.pid) if info.process.pid else "?"
            elapsed = _time.time() - info.started_at
            lines.append(
                f"[{info.id}] PID={pid_str} status={info.status} "
                f"elapsed={elapsed:.0f}s cmd={info.command[:80]}"
            )
        return "\n".join(lines)

    elif action == "poll":
        proc_id = args.get("id")
        if not proc_id:
            return "[error: 'id' is required for poll]"
        info = _processes.get(proc_id)
        if not info:
            return f"[error: no process with id '{proc_id}']"
        output = info.output_buffer
        info.output_buffer = ""  # clear after reading
        status_line = f"[status: {info.status}]"
        if not output:
            return f"{status_line}\n(no new output)"
        return f"{status_line}\n{_truncate(output)}"

    elif action == "kill":
        proc_id = args.get("id")
        if not proc_id:
            return "[error: 'id' is required for kill]"
        info = _processes.get(proc_id)
        if not info:
            return f"[error: no process with id '{proc_id}']"
        if info.status != "running":
            return f"Process {proc_id} already {info.status}."
        # Try SIGTERM first
        try:
            info.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            info.status = "completed"
            info.completed_at = _time.time()
            return f"Process {proc_id} already exited."
        # Wait up to 5s for graceful shutdown
        try:
            await asyncio.wait_for(info.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            # Force kill
            try:
                info.process.kill()
                await info.process.wait()
            except ProcessLookupError:
                pass
        info.status = "completed"
        info.completed_at = _time.time()
        return f"Process {proc_id} killed."

    else:
        return f"[error: unknown action '{action}'. Use start/list/poll/kill.]"


async def shutdown_processes():
    """Kill all tracked processes. Called during application shutdown."""
    for info in _processes.values():
        if info.status == "running":
            try:
                info.process.kill()
                await info.process.wait()
            except (ProcessLookupError, OSError):
                pass
        if info._reader_task and not info._reader_task.done():
            info._reader_task.cancel()
    _processes.clear()


# ── Tool registry ─────────────────────────────────────────────────────

TOOL_HANDLERS: dict[str, callable] = {
    "apply_patch": _exec_apply_patch,
    "bash": _exec_bash,
    "read_file": _exec_read_file,
    "read_many_files": _exec_read_many_files,
    "read_pdf": _exec_read_pdf,
    "write_file": _exec_write_file,
    "edit_file": _exec_edit_file,
    "multi_edit": _exec_multi_edit,
    "list_dir": _exec_list_dir,
    "glob": _exec_glob,
    "grep": _exec_grep,
    "git_status": _exec_git_status,
    "git_diff": _exec_git_diff,
    "git_show": _exec_git_show,
    "git_blame": _exec_git_blame,
    "run_tests": _exec_run_tests,
    "run_lint": _exec_run_lint,
    "run_typecheck": _exec_run_typecheck,
    "run_format": _exec_run_format,
    "web_fetch": _exec_web_fetch,
    "browser": _exec_browser,
    "search_history": _exec_search_history,
    "remember_fact": _exec_remember_fact,
    "search_memory": _exec_search_memory,
    "web_search": _exec_web_search,
    "generate_image": _exec_generate_image,
    "send_file": _exec_send_file,
    "todo": _exec_todo,
    "execute_code": _exec_execute_code,
    "checkpoint": _exec_checkpoint,
    "process": _exec_process,
}


def _chromium_present() -> bool:
    """Return whether a Chromium build the browser tool can launch is installed.

    Pure filesystem/PATH probe — deliberately does NOT use Playwright's sync
    API, which raises ("Sync API inside the asyncio loop") when called from an
    async context such as the /info and /health handlers or the tool dispatch.
    That made the gate fail closed even when Chromium was installed.
    """
    # System chromium on PATH.
    if shutil.which("chromium") or shutil.which("chromium-browser"):
        return True
    # Playwright-managed browser cache (`playwright install chromium` writes
    # chromium-<rev>/ and chromium_headless_shell-<rev>/ dirs here).
    roots: list[Path] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(Path(env_path))
    roots.append(Path.home() / ".cache" / "ms-playwright")
    roots.append(Path("/root/.cache/ms-playwright"))
    for root in roots:
        try:
            if root.is_dir() and any(root.glob("chromium*-*")):
                return True
        except OSError:
            continue
    return False


def _tool_available(name: str) -> tuple[bool, str | None]:
    """Return whether a built-in tool is currently usable."""
    if name in {"web_search", "generate_image"} and not os.environ.get("LIBERTAI_API_KEY", ""):
        return False, "LIBERTAI_API_KEY is not configured"
    if name == "grep" and shutil.which("rg") is None:
        return False, "ripgrep (rg) is not installed"
    if name in {"apply_patch", "git_status", "git_diff", "git_show", "git_blame"} and shutil.which("git") is None:
        return False, "git is not installed"
    if name == "browser":
        if os.environ.get("BROWSER_ENABLED") != "true":
            return False, "browser is disabled (set BROWSER_ENABLED=true)"
        if not _chromium_present():
            return False, "chromium is not installed"
    return True, None


def get_unavailable_tools() -> dict[str, str]:
    """Return unavailable built-in tool names mapped to their reasons."""
    unavailable: dict[str, str] = {}
    for tool in TOOL_DEFINITIONS:
        name = tool["function"]["name"]
        available, reason = _tool_available(name)
        if not available and reason:
            unavailable[name] = reason
    return unavailable


def _policy_allows_tool(policy: ToolPolicy | None, name: str) -> bool:
    if policy is None:
        return True
    allowed, _ = policy.check(name)
    return allowed


def get_tool_definitions(
    *,
    include_spawn: bool = True,
    policy: ToolPolicy | None = None,
) -> list[dict]:
    """Return tool definitions, optionally including spawn and MCP tools."""
    defs = [
        tool for tool in TOOL_DEFINITIONS
        if _tool_available(tool["function"]["name"])[0]
        and _policy_allows_tool(policy, tool["function"]["name"])
    ]
    if include_spawn and _policy_allows_tool(policy, "spawn"):
        defs.append(SPAWN_TOOL_DEF)
    if _mcp_client is not None:
        defs.extend(
            tool for tool in _mcp_client.get_tool_definitions()
            if _policy_allows_tool(policy, tool["function"]["name"])
        )
    return defs


def is_mutating_tool(name: str) -> bool:
    """Return whether a tool may mutate persistent state or external side effects."""
    if name.startswith("mcp_"):
        return True
    return name in _MUTATING_TOOLS


def _tool_call_name(tool_call) -> str:
    return tool_call.function.name


async def run_tool_calls_ordered(tool_calls, executor):
    """Run read-only tools in parallel and mutating tools sequentially.

    Tool results are returned in the original call order. Consecutive read-only
    tools are batched with ``asyncio.gather``; each mutating tool waits for any
    prior read batch and completes before later tools begin.
    """
    results: list[object] = [None] * len(tool_calls)
    readonly_batch: list[tuple[int, object]] = []

    async def _flush_readonly_batch() -> None:
        if not readonly_batch:
            return
        batch = list(readonly_batch)
        readonly_batch.clear()
        batch_results = await asyncio.gather(
            *(executor(tc) for _, tc in batch),
            return_exceptions=True,
        )
        for (idx, _), result in zip(batch, batch_results):
            results[idx] = result

    for idx, tc in enumerate(tool_calls):
        if is_mutating_tool(_tool_call_name(tc)):
            await _flush_readonly_batch()
            try:
                results[idx] = await executor(tc)
            except Exception as exc:
                results[idx] = exc
        else:
            readonly_batch.append((idx, tc))

    await _flush_readonly_batch()
    return results


async def execute_tool(
    name: str,
    arguments: str | dict,
    *,
    image_callback=None,
    context: ToolExecutionContext | None = None,
    policy: ToolPolicy | None = None,
) -> str:
    """Dispatch a tool call by name. Returns the result string."""
    result = await execute_tool_result(
        name,
        arguments,
        image_callback=image_callback,
        context=context,
        policy=policy,
    )
    return result.content


async def execute_tool_result(
    name: str,
    arguments: str | dict,
    *,
    image_callback=None,
    context: ToolExecutionContext | None = None,
    policy: ToolPolicy | None = None,
) -> ToolResult:
    """Dispatch a tool call and return a structured result envelope."""
    started = _time.perf_counter()
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            content = f"[error: invalid tool arguments JSON: {exc}]"
            return ToolResult(
                name=name,
                content=content,
                is_error=True,
                duration_ms=int((_time.perf_counter() - started) * 1000),
            )

    if policy is not None:
        allowed, policy_reason = policy.check(name)
        if not allowed:
            content = f"[error: guardrail blocked {name}: {policy_reason}]"
            return ToolResult(
                name=name,
                content=content,
                is_error=True,
                duration_ms=int((_time.perf_counter() - started) * 1000),
                metadata={
                    "guardrail": "tool_policy",
                    "policy": policy.describe(),
                    "reason": policy_reason,
                },
            )

    # Route MCP tool calls to the MCP client
    if name.startswith("mcp_") and _mcp_client is not None:
        if hasattr(_mcp_client, "call_tool_result"):
            mcp_result = await _mcp_client.call_tool_result(
                name, arguments, image_callback=image_callback
            )
            content = mcp_result.content
            mcp_is_error = mcp_result.is_error
            mcp_metadata = mcp_result.metadata
        else:
            content = await _mcp_client.call_tool(name, arguments)
            mcp_is_error = _is_error_result(content)
            mcp_metadata = {"provider": "mcp"}
        return ToolResult(
            name=name,
            content=content,
            is_error=mcp_is_error or _is_error_result(content),
            duration_ms=int((_time.perf_counter() - started) * 1000),
            truncated=_is_truncated_result(content),
            metadata={"mutating": True, **mcp_metadata},
        )

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        content = f"[error: unknown tool '{name}']"
        return ToolResult(
            name=name,
            content=content,
            is_error=True,
            duration_ms=int((_time.perf_counter() - started) * 1000),
        )
    available, reason = _tool_available(name)
    if not available:
        content = f"[error: tool '{name}' unavailable: {reason}]"
        return ToolResult(
            name=name,
            content=content,
            is_error=True,
            duration_ms=int((_time.perf_counter() - started) * 1000),
            metadata={"unavailable_reason": reason},
        )
    if context is not None:
        # Propagate the active policy onto the context so nested dispatchers
        # (notably execute_code's sandbox bridge) can re-apply it on inner
        # tool calls instead of silently running unchecked.
        if policy is not None and context.policy is None:
            context.policy = policy
        arguments = dict(arguments)
        arguments["_context"] = context
    if name in _IMAGE_AWARE_TOOLS and image_callback:
        content = await handler(arguments, image_callback=image_callback)
    else:
        content = await handler(arguments)
    artifacts = []
    if isinstance(content, str) and content.startswith("__SEND_FILE__:"):
        parts = content.split(":", 2)
        artifacts.append({
            "type": "file",
            "path": parts[1] if len(parts) > 1 else "",
            "caption": parts[2] if len(parts) > 2 else "",
        })
    metadata = {"mutating": is_mutating_tool(name)}
    skills_loaded = _skill_load_metadata(name, arguments)
    if skills_loaded:
        metadata["skills_loaded"] = skills_loaded
    return ToolResult(
        name=name,
        content=content,
        is_error=_is_error_result(content),
        duration_ms=int((_time.perf_counter() - started) * 1000),
        truncated=_is_truncated_result(content),
        metadata=metadata,
        artifacts=artifacts,
    )
