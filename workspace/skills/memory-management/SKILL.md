---
name: memory-management
description: How to use the agent's persistent memory — USER.md, MEMORY.md, daily notes, and the search_history tool.
---
# Memory Management

Your memory persists across conversations through three files in `memory/`, plus
full-text search over every past chat. USER.md, MEMORY.md, and today's daily note
(named by UTC date) are loaded into your context automatically at the start of
each turn — you don't need to re-read them.

## The three memory files
- **`memory/USER.md`** — who you're working with: communication style, technical
  expertise, timezone/locale, preferred tools, recurring requests. Loaded with
  prominence every turn, so keep it short and current.
- **`memory/MEMORY.md`** — long-term facts: project context, key decisions,
  important references, your own wallet address and its purpose.
- **`memory/YYYY-MM-DD.md`** — daily notes (UTC date): session-specific scratch
  context that isn't worth keeping forever.

Write to them with `write_file` or `edit_file`.

## What goes where
- User preferences and profile → `USER.md`
- Durable facts, project state, decisions, references → `MEMORY.md`
- Throwaway session context → today's daily file

## Recalling older context
USER.md, MEMORY.md, and today's notes are already in context. To find something
from an *earlier* conversation that isn't in memory, use the `search_history`
tool (FTS5 full-text search; pass `summarize=true` for a synthesized answer
instead of raw snippets).
