"""Context builder — assembles system prompt from memory, skills, and identity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_static_system_prompt(
    user_prompt: str,
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
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
        "- Save user preferences, project context, and important facts to MEMORY.md\n"
        "- Save session-specific notes to daily files\n"
        "- Read skill files for detailed instructions when a skill is relevant"
    )

    return "\n\n---\n\n".join(sections)


def build_dynamic_context(workspace_path: str) -> str:
    """Load memory and skills content for injection near end of message list.

    Kept separate from the static system prompt so the prefix tokens
    stay identical across turns, preserving the llama.cpp KV cache.
    """
    workspace = Path(workspace_path)
    sections = []

    memory = _load_memory(workspace)
    if memory:
        sections.append(f"## Memory\n\n{memory}")

    skills = _load_skills_summary(workspace)
    if skills:
        sections.append(f"## Available Skills\n\n{skills}")

    return "\n\n---\n\n".join(sections) if sections else ""


def build_system_prompt(
    user_prompt: str,
    agent_name: str,
    workspace_path: str,
    tool_names: list[str] | None = None,
) -> str:
    """Full system prompt (static + dynamic). Used for non-cached contexts."""
    static = build_static_system_prompt(user_prompt, agent_name, workspace_path, tool_names)
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


def _load_memory(workspace: Path) -> str:
    """Load MEMORY.md and today's daily notes."""
    parts = []

    # Long-term memory
    memory_file = workspace / "memory" / "MEMORY.md"
    if memory_file.exists():
        content = memory_file.read_text().strip()
        if content:
            parts.append(f"### Long-term Memory\n\n{content}")

    # Today's daily notes
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_file = workspace / "memory" / f"{today}.md"
    if daily_file.exists():
        content = daily_file.read_text().strip()
        if content:
            parts.append(f"### Today's Notes ({today})\n\n{content}")

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
        name, description = _parse_skill_metadata(
            skill_dir.name, skill_file.read_text()
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
