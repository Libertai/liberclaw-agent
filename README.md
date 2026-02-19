# LiberClaw Agent

Autonomous AI agent that runs on [Aleph Cloud](https://aleph.cloud) VMs, powered by [LibertAI](https://libertai.io) inference. Each agent is a FastAPI server with tool use, persistent memory, a skills system, and background subagents.

Part of the [LiberClaw](https://liberclaw.ai) platform.

## What it does

When deployed to a VM, the agent receives chat messages over SSE and can autonomously use tools to accomplish tasks. It maintains conversation history, compacts context when it grows too large, and remembers things across conversations via a file-based memory system.

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Shell execution with safety deny patterns and timeout |
| `read_file` | Read files with line numbers, offset/limit support |
| `write_file` | Write files, auto-create parent directories |
| `edit_file` | Find-and-replace (first occurrence) |
| `list_dir` | List directory contents |
| `web_fetch` | Fetch URLs, strip HTML, truncate large responses |
| `web_search` | Brave Search API (requires `BRAVE_API_KEY`) |
| `send_file` | Send workspace files back to the user |
| `spawn` | Launch background subagents for parallel work |

## Features

- **Agentic loop**: Tool calls are executed in a loop until the model produces a final text response or hits the iteration limit
- **SSE streaming**: Chat responses stream as server-sent events with keepalives to survive reverse proxies
- **Context compaction**: When conversation history exceeds the model's context window, older messages are summarized by the LLM while recent messages are preserved
- **Memory**: Persistent file-based memory at `workspace/memory/MEMORY.md` (long-term) and daily notes
- **Skills**: Markdown skill files at `workspace/skills/*/SKILL.md` — summaries are injected into context, full content loaded on demand
- **Subagents**: Spawn background workers for parallel tasks with their own tool sets (no further spawning)
- **Heartbeat**: Periodic check of `HEARTBEAT.md` for autonomous task execution
- **Security**: Bearer token auth, workspace path sandboxing, bash command deny patterns, sensitive file protection

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, receive SSE stream |
| `/chat/{chat_id}/history` | GET | Retrieve conversation history |
| `/chat/{chat_id}` | DELETE | Clear conversation history |
| `/pending` | GET | Poll for proactive messages (heartbeat, subagent results) |
| `/files/{path}` | GET | Download a workspace file |
| `/files/upload` | POST | Upload a file to the workspace |
| `/workspace/tree` | GET | Recursive workspace file tree |
| `/subagents` | GET | List all subagent runs |
| `/subagents/{id}` | GET | Get subagent details |
| `/subagents/{id}/stop` | POST | Cancel a running subagent |
| `/health` | GET | Health check (no auth required) |

All endpoints except `/health` require `Authorization: Bearer <agent_secret>`.

## Running locally

```bash
pip install fastapi uvicorn pydantic-settings openai httpx aiosqlite

AGENT_NAME=test \
SYSTEM_PROMPT="You are a helpful assistant." \
MODEL=qwen3-coder-next \
LIBERTAI_API_KEY=your-key \
AGENT_SECRET=your-secret \
WORKSPACE_PATH=/tmp/agent-workspace \
  uvicorn baal_agent.main:app --port 8080
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_NAME` | `Agent` | Display name |
| `SYSTEM_PROMPT` | `You are a helpful assistant.` | Custom instructions |
| `MODEL` | — | LibertAI model to use |
| `LIBERTAI_API_KEY` | — | API key for inference |
| `AGENT_SECRET` | — | Bearer token for auth |
| `WORKSPACE_PATH` | `/opt/baal-agent/workspace` | Root directory for files |
| `OWNER_CHAT_ID` | — | Chat ID for heartbeat delivery |
| `HEARTBEAT_INTERVAL` | `1800` | Seconds between heartbeat checks (0 to disable) |
| `MAX_TOOL_ITERATIONS` | `50` | Max tool calls per turn |
| `MAX_CONTEXT_TOKENS` | `0` | Context limit (0 = auto-detect from model) |
| `INFERENCE_TIMEOUT` | `180` | Seconds before inference timeout |
| `BRAVE_API_KEY` | — | Enables `web_search` tool |

## License

AGPL-3.0
