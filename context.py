"""Context builder — assembles system prompt from memory, skills, and identity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_static_system_prompt(
    user_prompt: str,
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
    heartbeat_interval: int = 0,
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
    if tool_names:
        identity += f"\nAvailable tools: {', '.join(tool_names)}"
    sections.append(identity)

    # User's custom instructions
    if user_prompt and user_prompt.strip():
        sections.append(f"## Instructions\n\n{user_prompt.strip()}")

    # Memory system instructions
    sections.append(
        "## Memory System\n\n"
        "You have persistent memory. To remember things across conversations:\n"
        f"- Long-term: Write to `{workspace_path}/memory/MEMORY.md` using write_file or edit_file\n"
        f"- Daily notes: Write to `{workspace_path}/memory/{today}.md`\n"
        f"- User profile: Write to `{workspace_path}/memory/USER.md`\n"
        "- Save project context and important facts to MEMORY.md\n"
        "- Save session-specific notes to daily files\n"
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


def build_dynamic_context(workspace_path: str) -> str:
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

    skills = _load_skills_summary(workspace)
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
) -> str:
    """Full system prompt (static + dynamic). Used for non-cached contexts."""
    static = build_static_system_prompt(
        user_prompt, agent_name, workspace_path, tool_names,
        heartbeat_interval=heartbeat_interval,
    )
    dynamic = build_dynamic_context(workspace_path)
    if dynamic:
        return static + "\n\n---\n\n" + dynamic
    return static


def build_subagent_prompt(
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
    persona: str | None = None,
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

    # Optional persona / role
    if persona and persona.strip():
        sections.append(f"## Role\n\n{persona.strip()}")

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
            content = user_file.read_text().strip()
            if content:
                return content
    except (OSError, UnicodeDecodeError):
        pass  # Corrupted or unreadable file — skip silently
    return ""


_CONTEXT_FILENAMES = ("CONTEXT.md", "AGENTS.md", ".hermes.md", "CLAUDE.md")
_CONTEXT_MAX_CHARS = 20_000


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
            content = path.read_text().strip()
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
            content = memory_file.read_text().strip()
            if content:
                parts.append(f"### Long-term Memory\n\n{content}")
    except (OSError, UnicodeDecodeError):
        pass

    # Today's daily notes
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_file = workspace / "memory" / f"{today}.md"
    try:
        if daily_file.exists():
            content = daily_file.read_text().strip()
            if content:
                parts.append(f"### Today's Notes ({today})\n\n{content}")
    except (OSError, UnicodeDecodeError):
        pass

    return "\n\n".join(parts)


def _load_skills_summary(workspace: Path) -> str:
    """Scan workspace/skills/*/SKILL.md and return a summary list."""
    skills_dir = workspace / "skills"
    if not skills_dir.is_dir():
        return ""

    lines = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            skill_content = skill_file.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        name, description = _parse_skill_metadata(
            skill_dir.name, skill_content
        )
        lines.append(f"- **{name}**: {description} (read `{skill_file}` for details)")

    return "\n".join(lines) if lines else ""


def _parse_skill_metadata(dir_name: str, content: str) -> tuple[str, str]:
    """Extract skill name and description from SKILL.md content.

    Supports both agentskills.io format (YAML frontmatter with name/description)
    and legacy format (# Title followed by first non-heading paragraph).
    """
    lines = content.splitlines()

    # Detect YAML frontmatter (--- delimited block at the start)
    if lines and lines[0].strip() == "---":
        fm_name = ""
        fm_description = ""
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                # End of frontmatter — use parsed values if we got a description
                if fm_description:
                    return fm_name or dir_name, fm_description
                # No description in frontmatter, fall through to legacy parsing
                # but skip past the frontmatter block
                return _parse_legacy_description(dir_name, lines[i + 1 :])
            stripped = line.strip()
            if stripped.startswith("name:"):
                fm_name = stripped[5:].strip().strip("\"'")
            elif stripped.startswith("description:"):
                fm_description = stripped[12:].strip().strip("\"'")

    # No frontmatter — legacy format
    return _parse_legacy_description(dir_name, lines)


def _parse_legacy_description(
    dir_name: str, lines: list[str]
) -> tuple[str, str]:
    """Extract description as the first non-empty, non-heading line."""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return dir_name, stripped
    return dir_name, ""
