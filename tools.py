"""Tool definitions and executors for agent VMs."""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import re
import signal
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
from baal_agent.pii import redact_pii
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

_IMAGE_AWARE_TOOLS = {"read_file", "read_pdf", "web_fetch"}

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
            "description": "Find and replace an exact string in a file (first occurrence).",
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
                },
                "required": ["path", "old_string", "new_string"],
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
            lines = f.readlines()
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
        if old_string not in content:
            return f"[error: old_string not found in {path}]"
        content = content.replace(old_string, new_string, 1)
        with open(resolved, "w") as f:
            f.write(content)
        return f"Edited {path}"
    except PathSecurityError as e:
        return f"[error: {e}]"
    except FileNotFoundError:
        return f"[error: file not found: {path}]"
    except Exception as e:
        return f"[error: {e}]"


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
    return await _code_executor.execute(code, timeout=timeout)


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
    "bash": _exec_bash,
    "read_file": _exec_read_file,
    "read_pdf": _exec_read_pdf,
    "write_file": _exec_write_file,
    "edit_file": _exec_edit_file,
    "list_dir": _exec_list_dir,
    "web_fetch": _exec_web_fetch,
    "search_history": _exec_search_history,
    "web_search": _exec_web_search,
    "generate_image": _exec_generate_image,
    "send_file": _exec_send_file,
    "todo": _exec_todo,
    "execute_code": _exec_execute_code,
    "checkpoint": _exec_checkpoint,
    "process": _exec_process,
}


def get_tool_definitions(*, include_spawn: bool = True) -> list[dict]:
    """Return tool definitions, optionally including spawn and MCP tools."""
    defs = list(TOOL_DEFINITIONS)
    if include_spawn:
        defs.append(SPAWN_TOOL_DEF)
    if _mcp_client is not None:
        defs.extend(_mcp_client.get_tool_definitions())
    return defs


async def execute_tool(
    name: str,
    arguments: str | dict,
    *,
    image_callback=None,
    pii_redaction: bool = True,
) -> str:
    """Dispatch a tool call by name. Returns the result string.

    When *pii_redaction* is True (the default), the result is scanned for
    sensitive data patterns (credit cards, SSNs, API keys, emails) and
    matches are replaced with ``[REDACTED:type]`` tokens.  Code blocks
    inside fenced ``` markers are left untouched.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    # Route MCP tool calls to the MCP client
    if name.startswith("mcp_") and _mcp_client is not None:
        result = await _mcp_client.call_tool(name, arguments)
        if pii_redaction:
            result = redact_pii(result)
        return result

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"[error: unknown tool '{name}']"
    if name in _IMAGE_AWARE_TOOLS and image_callback:
        result = await handler(arguments, image_callback=image_callback)
    else:
        result = await handler(arguments)
    if pii_redaction:
        result = redact_pii(result)
    return result
