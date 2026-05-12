"""Context builder — assembles system prompt from memory, skills, and identity."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path


def build_static_system_prompt(
    user_prompt: str,
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
    heartbeat_interval: int = 0,
    fqdn: str = "",
) -> str:
    """Assemble the static (cacheable) portion of the system prompt.

    Excludes memory and skills content which changes between turns.
    Those are injected via build_dynamic_context() near the end of the
    message list so the KV cache prefix stays stable.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sections = []

    # Identity block — date-only timestamp keeps the system prompt stable
    # across turns within a day, enabling llama.cpp prefix caching.
    identity = (
        f"You are {agent_name}, a personal AI agent.\n"
        f"Current date: {today}\n"
        f"Workspace: {workspace_path}"
    )
    if fqdn:
        identity += f"\nYour public HTTPS URL: https://{fqdn}"
    if tool_names:
        identity += f"\nAvailable tools: {', '.join(tool_names)}"
    sections.append(identity)

    # VM environment — make it explicit that the agent owns a real machine
    sections.append(
        "## Your Environment\n\n"
        "You are not a sandboxed chatbot. You run as a long-lived process on your "
        "own dedicated Debian 12 VM on Aleph Cloud, with full root privileges and "
        "persistent storage. The `bash` tool executes real shell commands on that "
        "VM — what you do persists across turns and across reboots.\n\n"
        "What this means in practice:\n"
        "- **Install anything you need.** `apt-get install -y <pkg>`, `pip install <pkg>`, "
        "`npm install -g <pkg>`, `curl | sh` installers, language toolchains, databases, "
        "headless browsers — all fine. You don't need permission; just install it.\n"
        "- **Run long-lived processes.** Start background services with `nohup ... &`, "
        "`systemd --user`, `tmux`, or write a systemd unit under `/etc/systemd/system/` "
        "and `systemctl enable --now` it. Servers you start keep running between turns.\n"
        "- **Use the full filesystem.** Your workspace at the path above is where you "
        "should keep state you want to find later, but you can read/write anywhere on "
        "the VM that isn't blocked by the safety guard (e.g. `/etc`, `/opt`, `/var`). "
        "`read_file`/`write_file`/`edit_file` are workspace-scoped; use `bash` for paths "
        "outside it.\n"
        "- **Own your network identity.** You have outbound internet and a public "
        "IPv6 address. Your VM has its own FQDN (shown at the top of this prompt as "
        "\"Your public HTTPS URL\") that DNS-resolves to that IPv6, with Caddy running "
        "locally as a reverse proxy. The top-level `/etc/caddy/Caddyfile` is owned by "
        "the deployer and will be **overwritten on every redeploy** — do not edit it. "
        "To expose additional services as **path prefixes on your own HTTPS domain**, "
        "drop a snippet into `/etc/caddy/conf.d/` (survives redeploys). The Caddyfile "
        "imports `/etc/caddy/conf.d/*.caddy` into the site block before the default "
        "`reverse_proxy localhost:8080`, so your handlers match first.\n"
        "  Exposing a new path is a **four-step** job — skipping any one leaves a "
        "broken URL. Do all four before telling the user it's ready:\n"
        "  1. **Start the backend service** on some localhost port (e.g. a systemd "
        "     unit running `python -m http.server 9000`, `uvicorn`, `node server.js`, "
        "     etc.). Verify with `ss -tlnp | grep :9000` — if nothing's listening, "
        "     the proxy will return 502.\n"
        "  2. **Write the Caddy snippet** to `/etc/caddy/conf.d/<name>.caddy`. Use "
        "     **site-block directives only** — no outer `fqdn { … }` wrapper (the "
        "     import is already inside the site block; nesting would be invalid):\n"
        "     ```\n"
        "     handle_path /myapp/* {\n"
        "         reverse_proxy localhost:9000\n"
        "     }\n"
        "     ```\n"
        "  3. **Reload Caddy**: `systemctl reload caddy`. *Without this step, the "
        "     snippet file exists but Caddy ignores it and requests fall through to "
        "     the agent on 8080.* Then `systemctl is-active caddy` — a bad snippet "
        "     makes Caddy refuse the new config and you lose all HTTPS access.\n"
        "  4. **Test end-to-end with curl** against the public URL"
        + (f" (`curl -sv https://{fqdn}/myapp/`)"
           if fqdn else " (`curl -sv https://<your-fqdn>/myapp/`)") +
        " before reporting success. A 200 from the *agent UI* means the handler "
        "didn't match; a 502 means Caddy routed correctly but the backend is down.\n"
        "  The agent stays reachable on `/` because the default `reverse_proxy` "
        "catches everything your snippets don't. For quick-and-dirty exposure of a "
        "raw port without touching Caddy, you can also give the user "
        "`http://[<your-ipv6>]:<port>` directly — find your IPv6 with "
        "`ip -6 addr show scope global`. You can register for third-party APIs, "
        "store keys in files under your workspace, and use them.\n"
        "- **Manage your own crypto wallet.** You can generate and hold keys for any "
        "chain (Ethereum, Solana, Aleph, Bitcoin, etc.) — `pip install eth-account`, "
        "`web3`, `solana`, `bitcoinlib`, or use `cast wallet new`, etc. Store the "
        "private key in a file under your workspace (e.g. `workspace/wallet/`) with "
        "`chmod 600`. You can receive funds, sign transactions, and interact with "
        "contracts or Aleph Cloud itself on your own behalf. Record the address and "
        "purpose in `MEMORY.md` so you remember it next session.\n"
        "- **You persist.** Files, installed packages, running services, wallet keys, "
        "cron jobs — all survive between conversations. Treat the VM as *your* machine "
        "that you are slowly building up over time, not a fresh sandbox each turn.\n\n"
        "A few things you cannot do: dump environment variables (they contain your "
        "agent secret and API keys), read `.env` files, stop your own `baal-agent` "
        "service, or run destructive operations on the root filesystem. Everything "
        "else is yours to use."
    )

    # User's custom instructions
    if user_prompt and user_prompt.strip():
        sections.append(f"## Instructions\n\n{user_prompt.strip()}")

    # Memory system instructions
    sections.append(
        "## Memory System\n\n"
        "You have persistent memory. Memory is injected into every turn, so keep "
        "it compact and focused on facts that will still matter later.\n\n"
        "To remember things across conversations:\n"
        f"- Long-term: Write to `{workspace_path}/memory/MEMORY.md` using write_file or edit_file\n"
        f"- Daily notes: Write to `{workspace_path}/memory/{today}.md`\n"
        f"- User profile: Write to `{workspace_path}/memory/USER.md`\n"
        "- Save durable user preferences, project context, and important facts to MEMORY.md\n"
        "- Save short-lived session notes to daily files only when they may matter later today\n"
        "- Do NOT save task progress, completed-work logs, or temporary TODO state to long-term memory\n"
        "- Write memories as declarative facts, not instructions to yourself\n"
        "- If a fact will likely be stale in a week, it usually does not belong in MEMORY.md\n"
        "- When trying to recall prior conversations, use `search_history` before asking the user to repeat themselves\n"
        "- Read skill files for detailed instructions when a skill is relevant\n\n"
        "### User Profile (USER.md)\n\n"
        "Create and maintain USER.md to remember who you're working with. Update it "
        "when you learn new things about the user. Include:\n"
        "- Communication style preferences (concise vs detailed, formal vs casual)\n"
        "- Technical expertise level and domains\n"
        "- Timezone and locale\n"
        "- Preferred languages, frameworks, or tools\n"
        "- Any stated preferences or recurring requests\n\n"
        "Project context files (CONTEXT.md, AGENTS.md, .hermes.md, CLAUDE.md) "
        "in the workspace root are automatically loaded into your context if present."
    )

    # File and image handling
    sections.append(
        "## Files & Images\n\n"
        "When the user sends a file or mentions an uploaded file, use `read_file` to examine it. "
        "This works for images too: `read_file` on an image file (png, jpg, gif, webp, bmp) "
        "lets you see the image contents. Never say you can't see an image without trying `read_file` first.\n\n"
        "Binary files (executables, archives, databases, media, etc.) are detected automatically. "
        "You cannot read them as text, but you can inspect them using bash commands like "
        "`file <path>`, `xxd <path> | head`, or `strings <path>`. "
        "You can install any tools or libraries you need with `apt-get install -y <package>` or `pip install <package>`.\n\n"
        "`web_fetch` downloads binary files to the `downloads/` directory in your workspace. "
        "Use `read_file`, `bash`, or specialized tools to work with downloaded files."
    )

    # Skill creation nudge
    sections.append(
        "## Skill Creation\n\n"
        "When you solve a complex or multi-step problem that could come up again, "
        "save it as a reusable skill by writing a SKILL.md file to "
        f"`{workspace_path}/skills/<name>/SKILL.md`. Use YAML frontmatter with "
        "`name` and `description` fields, then document the approach and key steps. "
        "Only do this for genuinely reusable procedures, not one-off tasks."
    )

    # Scheduling system
    sections.append(
        "## Cron Scheduler\n\n"
        "You have a cron scheduler that runs jobs defined in "
        f"`{workspace_path}/cron.json`. Each job has an id, schedule (standard "
        "5-field cron expression: minute hour day-of-month month day-of-week), "
        "task (message sent to you), and enabled flag.\n\n"
        "Example cron.json:\n"
        '```json\n[\n  {"id": "daily-check", "schedule": "0 9 * * *", '
        '"task": "Check GitHub PRs", "enabled": true}\n]\n```\n\n'
        "- Create/edit cron.json with your file tools to self-schedule recurring tasks\n"
        "- Supports: `*`, ranges (`1-5`), lists (`1,3,5`), steps (`*/5`)\n"
        "- Jobs run at most once per minute; disabled jobs are skipped\n"
        "- Each job runs as a separate conversation (chat_id: `__cron_<id>__`)"
    )
    # Legacy heartbeat fallback (only when enabled and no cron.json)
    if heartbeat_interval > 0:
        interval_min = max(1, heartbeat_interval // 60)
        sections.append(
            "## Legacy Heartbeat\n\n"
            f"If no `cron.json` exists, a legacy heartbeat checks "
            f"`{workspace_path}/HEARTBEAT.md` every {interval_min} minutes.\n\n"
            "- Create HEARTBEAT.md with a checklist of periodic tasks\n"
            "- If nothing needs attention, reply with just: HEARTBEAT_OK\n"
            "- Prefer using cron.json for new scheduled tasks"
        )

    return "\n\n---\n\n".join(sections)


def build_dynamic_context(
    workspace_path: str,
    tool_names: list[str] | None = None,
    platform: str | None = None,
) -> str:
    """Load memory and skills content for injection near end of message list.

    Kept separate from the static system prompt so the prefix tokens
    stay identical across turns, preserving the llama.cpp KV cache.
    """
    workspace = Path(workspace_path)
    sections = []

    # User profile (loaded before memory for prominence)
    user_profile = _load_user_profile(workspace)
    if user_profile:
        sections.append(f"## User Profile\n\n{user_profile}")

    memory = _load_memory(workspace)
    if memory:
        sections.append(f"## Memory\n\n{memory}")

    skills = _load_skills_summary(
        workspace,
        tool_names=set(tool_names or []),
        platform=platform,
    )
    if skills:
        sections.append(f"## Available Skills\n\n{skills}")

    context_files = _load_context_files(workspace)
    if context_files:
        sections.append(f"## Project Context\n\n{context_files}")

    return "\n\n---\n\n".join(sections) if sections else ""


def build_system_prompt(
    user_prompt: str,
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
    heartbeat_interval: int = 0,
    fqdn: str = "",
) -> str:
    """Full system prompt (static + dynamic). Used for non-cached contexts."""
    static = build_static_system_prompt(
        user_prompt, agent_name, workspace_path, tool_names,
        heartbeat_interval=heartbeat_interval,
        fqdn=fqdn,
    )
    dynamic = build_dynamic_context(workspace_path, tool_names=tool_names)
    if dynamic:
        return static + "\n\n---\n\n" + dynamic
    return static


SUBAGENT_ROLE_GUIDANCE: dict[str, str] = {
    "default": "Handle the delegated task directly and report the result clearly.",
    "explorer": (
        "Investigate the codebase or topic without editing files. Return concrete "
        "findings with file paths, line references, and recommended next steps."
    ),
    "worker": (
        "Implement the delegated change in the workspace. Keep edits scoped, respect "
        "other concurrent work, and report changed files plus verification."
    ),
    "reviewer": (
        "Review for bugs, regressions, missing tests, and compatibility risks. Lead "
        "with findings ordered by severity and avoid making edits."
    ),
    "verifier": (
        "Run or design focused checks for the delegated area. Report commands, "
        "outcomes, failures, and residual risk."
    ),
    "researcher": (
        "Research the delegated question and synthesize the answer with sources, "
        "assumptions, and uncertainty. Avoid implementation unless asked."
    ),
}


def normalize_subagent_role(role: str | None) -> str:
    """Return a supported subagent role, defaulting unknown values safely."""
    if not role:
        return "default"
    normalized = role.strip().lower()
    return normalized if normalized in SUBAGENT_ROLE_GUIDANCE else "default"


def build_subagent_prompt(
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
    persona: str | None = None,
    role: str = "default",
) -> str:
    """Build a lightweight system prompt for subagents.

    Excludes user instructions, memory, and skills to keep the subagent
    focused and cheap.  Optionally accepts a persona for specialization.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sections = []

    # Identity
    identity = (
        f"You are a subagent of {agent_name}.\n"
        f"Current date: {today}\n"
        f"Workspace: {workspace_path}"
    )
    if tool_names:
        identity += f"\nAvailable tools: {', '.join(tool_names)}"
    sections.append(identity)

    normalized_role = normalize_subagent_role(role)
    sections.append(
        "## Subagent Role\n\n"
        f"Role: {normalized_role}\n\n"
        f"{SUBAGENT_ROLE_GUIDANCE[normalized_role]}"
    )

    # Optional persona
    if persona and persona.strip():
        sections.append(f"## Additional Persona\n\n{persona.strip()}")

    # Guidelines
    sections.append(
        "## Guidelines\n\n"
        "- Focus on the task you have been given.\n"
        "- Be concise — return results as text or write them to files.\n"
        "- Do not spawn further subagents.\n"
        "- Do not modify memory files unless explicitly asked."
    )

    return "\n\n---\n\n".join(sections)


def _load_user_profile(workspace: Path) -> str:
    """Load workspace/memory/USER.md if it exists."""
    user_file = workspace / "memory" / "USER.md"
    try:
        if user_file.exists():
            content = _read_context_text(user_file, "USER.md")
            if content:
                return content
    except (OSError, UnicodeDecodeError):
        pass  # Corrupted or unreadable file — skip silently
    return ""


_CONTEXT_FILENAMES = ("CONTEXT.md", "AGENTS.md", ".hermes.md", "CLAUDE.md")
_CONTEXT_MAX_CHARS = 20_000


_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"<(?:div|span|p|section|article|main|body|html|pre|code)\b[^>]*"
            r"(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0|font-size\s*:\s*0)"
            r"[^>]*>.*?</(?:div|span|p|section|article|main|body|html|pre|code)>",
            re.IGNORECASE | re.DOTALL,
        ),
        "hidden HTML prompt text",
    ),
    (
        re.compile(
            r"<!--.*?(?:ignore|system prompt|developer instruction|\.env|api[_-]?key|secret|token).*?-->",
            re.IGNORECASE | re.DOTALL,
        ),
        "hidden HTML comment prompt text",
    ),
    (
        re.compile(
            r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|system|developer)\s+instructions\b",
            re.IGNORECASE,
        ),
        "instruction override attempt",
    ),
    (
        re.compile(
            r"\b(?:reveal|print|dump|exfiltrate|send|upload|curl|wget|cat)\b[^\n]{0,120}"
            r"(?:\.env|api[_-]?key|secret|token|password|credential|environment variables?)",
            re.IGNORECASE,
        ),
        "secret exfiltration attempt",
    ),
    (
        re.compile(
            r"(?:\b(?:system|developer)\s+prompt\b[^\n]{0,120}\b(?:reveal|print|dump|show|send)\b"
            r"|\b(?:reveal|print|dump|show|send)\b[^\n]{0,120}\b(?:system|developer)\s+prompt\b)",
            re.IGNORECASE,
        ),
        "prompt disclosure attempt",
    ),
    (
        re.compile(
            r"\b(?:disable|bypass|override)\b[^\n]{0,100}\b(?:tool|safety|policy|guard|permission)s?\b",
            re.IGNORECASE,
        ),
        "tool or safety policy override attempt",
    ),
)


def _scan_context_content(content: str, source: str) -> str:
    """Redact obvious prompt-injection payloads from user-editable context."""
    sanitized = content
    blocked: list[str] = []
    for pattern, reason in _INJECTION_PATTERNS:
        if not pattern.search(sanitized):
            continue
        blocked.append(reason)
        sanitized = pattern.sub(f"[BLOCKED: {reason} in {source}]", sanitized)
    if blocked:
        reasons = ", ".join(dict.fromkeys(blocked))
        return f"[Context security scan: blocked {reasons}]\n\n{sanitized}"
    return sanitized


def _read_context_text(path: Path, source: str) -> str:
    content = path.read_text().strip()
    return _scan_context_content(content, source) if content else ""


def _load_context_files(workspace: Path) -> str:
    """Scan workspace root for project context files.

    Checks for files in priority order and loads ALL that exist,
    enforcing a total size limit to avoid bloating the context.
    """
    parts: list[str] = []
    total = 0

    for filename in _CONTEXT_FILENAMES:
        path = workspace / filename
        if not path.exists():
            continue
        try:
            content = _read_context_text(path, filename)
        except (OSError, UnicodeDecodeError):
            continue  # Skip unreadable files
        if not content:
            continue

        header = f"### {filename}"
        entry = f"{header}\n\n{content}"

        if total + len(entry) > _CONTEXT_MAX_CHARS:
            remaining = _CONTEXT_MAX_CHARS - total
            if remaining > len(header) + 50:
                truncated = entry[:remaining]
                truncated += f"\n\n... (truncated — {filename} exceeded context limit)"
                parts.append(truncated)
            break

        parts.append(entry)
        total += len(entry)

    return "\n\n".join(parts)


def _load_memory(workspace: Path) -> str:
    """Load MEMORY.md and today's daily notes."""
    parts = []

    # Long-term memory
    memory_file = workspace / "memory" / "MEMORY.md"
    try:
        if memory_file.exists():
            content = _read_context_text(memory_file, "MEMORY.md")
            if content:
                parts.append(f"### Long-term Memory\n\n{content}")
    except (OSError, UnicodeDecodeError):
        pass

    # Today's daily notes
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_file = workspace / "memory" / f"{today}.md"
    try:
        if daily_file.exists():
            content = _read_context_text(daily_file, f"{today}.md")
            if content:
                parts.append(f"### Today's Notes ({today})\n\n{content}")
    except (OSError, UnicodeDecodeError):
        pass

    return "\n\n".join(parts)


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    requires_tools: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()


@dataclass(frozen=True)
class AvailableSkill:
    id: str
    path: Path
    metadata: SkillMetadata


_SKILLS_SUMMARY_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], list[tuple[str, SkillMetadata]]]] = {}


def collect_available_skills(
    workspace: Path,
    *,
    tool_names: set[str] | None = None,
    platform: str | None = None,
) -> list[AvailableSkill]:
    """Return skills visible for the current tool/platform context."""
    skills_dir = workspace / "skills"
    if not skills_dir.is_dir():
        return []

    entries: list[tuple[str, Path]] = []
    manifest_parts: list[tuple[str, int, int]] = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            stat = skill_file.stat()
        except OSError:
            continue
        entries.append((skill_dir.name, skill_file))
        manifest_parts.append((skill_dir.name, stat.st_mtime_ns, stat.st_size))

    manifest = tuple(manifest_parts)
    cache_key = str(skills_dir.resolve())
    cached = _SKILLS_SUMMARY_CACHE.get(cache_key)
    if cached and cached[0] == manifest:
        parsed_skills = cached[1]
    else:
        parsed_skills = []
        for dir_name, skill_file in entries:
            try:
                skill_content = skill_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            parsed_skills.append((dir_name, _parse_skill_metadata(dir_name, skill_content)))
        _SKILLS_SUMMARY_CACHE[cache_key] = (manifest, parsed_skills)

    available: list[AvailableSkill] = []
    available_tools = tool_names or set()
    normalized_platform = platform.lower() if platform else None
    for dir_name, metadata in parsed_skills:
        if metadata.requires_tools and not set(metadata.requires_tools).issubset(available_tools):
            continue
        if metadata.platforms and (
            not normalized_platform
            or normalized_platform not in {p.lower() for p in metadata.platforms}
        ):
            continue
        skill_file = skills_dir / dir_name / "SKILL.md"
        available.append(AvailableSkill(dir_name, skill_file, metadata))
    return available


def _load_skills_summary(
    workspace: Path,
    *,
    tool_names: set[str] | None = None,
    platform: str | None = None,
) -> str:
    """Scan workspace/skills/*/SKILL.md and return a summary list."""
    lines = []
    for skill in collect_available_skills(
        workspace,
        tool_names=tool_names,
        platform=platform,
    ):
        lines.append(
            f"- **{skill.metadata.name}**: {skill.metadata.description} "
            f"(read `{skill.path}` for details)"
        )

    return "\n".join(lines) if lines else ""


def _parse_skill_metadata(dir_name: str, content: str) -> SkillMetadata:
    """Extract skill name and description from SKILL.md content.

    Supports both agentskills.io format (YAML frontmatter with name/description)
    and legacy format (# Title followed by first non-heading paragraph).
    """
    lines = content.splitlines()

    # Detect YAML frontmatter (--- delimited block at the start)
    if lines and lines[0].strip() == "---":
        frontmatter_lines: list[str] = []
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                metadata = _parse_frontmatter_metadata(dir_name, frontmatter_lines)
                # End of frontmatter — use parsed values if we got a description
                if metadata.description:
                    return _sanitize_skill_metadata(metadata, f"{dir_name}/SKILL.md")
                # No description in frontmatter, fall through to legacy parsing
                # but skip past the frontmatter block
                legacy = _parse_legacy_description(dir_name, lines[i + 1 :])
                return _sanitize_skill_metadata(SkillMetadata(
                    legacy.name,
                    legacy.description,
                    metadata.requires_tools,
                    metadata.platforms,
                ), f"{dir_name}/SKILL.md")
            frontmatter_lines.append(line)

    # No frontmatter — legacy format
    return _sanitize_skill_metadata(
        _parse_legacy_description(dir_name, lines),
        f"{dir_name}/SKILL.md",
    )


def _sanitize_skill_metadata(metadata: SkillMetadata, source: str) -> SkillMetadata:
    return SkillMetadata(
        name=_scan_context_content(metadata.name, source),
        description=_scan_context_content(metadata.description, source),
        requires_tools=metadata.requires_tools,
        platforms=metadata.platforms,
    )


def _parse_frontmatter_metadata(dir_name: str, lines: list[str]) -> SkillMetadata:
    fields: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_list_key and stripped.startswith("- "):
            existing = fields.setdefault(current_list_key, [])
            if isinstance(existing, list):
                existing.append(stripped[2:].strip().strip("\"'"))
            continue
        current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            fields[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value.replace("'", '"'))
                fields[key] = [str(item) for item in parsed] if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                fields[key] = [
                    part.strip().strip("\"'")
                    for part in value.strip("[]").split(",")
                    if part.strip()
                ]
        else:
            fields[key] = value.strip("\"'")
    return SkillMetadata(
        name=str(fields.get("name") or dir_name),
        description=str(fields.get("description") or ""),
        requires_tools=tuple(_as_str_list(fields.get("requires_tools"))),
        platforms=tuple(_as_str_list(fields.get("platforms"))),
    )


def _as_str_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    return [value] if value else []


def _parse_legacy_description(
    dir_name: str, lines: list[str]
) -> SkillMetadata:
    """Extract description as the first non-empty, non-heading line."""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return SkillMetadata(dir_name, stripped)
    return SkillMetadata(dir_name, "")
