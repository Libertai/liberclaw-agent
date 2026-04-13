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
| `web_search` | LibertAI Search (always available) |
| `generate_image` | Generate images from text prompts via LibertAI |
| `send_file` | Send workspace files back to the user |
| `spawn` | Launch background subagents for parallel work |

## Features

- **Agentic loop**: Tool calls are executed in a loop until the model produces a final text response or hits the iteration limit
- **SSE streaming**: Chat responses stream as server-sent events with keepalives to survive reverse proxies
- **Context compaction**: When conversation history exceeds the model's context window, older messages are summarized by the LLM while recent messages are preserved
- **Memory**: Persistent file-based memory at `workspace/memory/MEMORY.md` (long-term) and daily notes
- **Skills**: Markdown skill files at `workspace/skills/*/SKILL.md` — summaries are injected into context, full content loaded on demand
- **Subagents**: Spawn background workers for parallel tasks with their own tool sets (no further spawning)
- **Heartbeat**: Periodic check of `workspace/HEARTBEAT.md` for autonomous task execution (configurable interval, results stored in chat history)
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

## Local Web UI

The agent serves a browser chat UI at `/` (provided `sites/agent-ui/` has been built). Users point their browser at the agent's HTTPS URL (for LiberClaw-deployed agents) or `http://localhost:8080` (for standalone installs) and log in with their `AGENT_SECRET`.

LiberClaw users can click the "Open direct chat" button on the agent detail screen to open the UI in a new tab with the token pre-filled in the URL fragment (`https://<agent-fqdn>/#token=<secret>&chat=<chat_id>`). The fragment is scrubbed by the SPA's bootstrap code before React mounts, so the secret never reaches server logs.

### Building the SPA

The SPA is built from `sites/agent-ui/` (in the parent repo) into `src/baal_agent/webui/dist/`. For standalone installs:

```bash
cd /path/to/baal
cd sites/agent-ui
npm install
npm run build
```

For LiberClaw-orchestrated deploys, the build runs automatically as a pre-deploy hook (see `src/liberclaw/services/agent_manager.py:ensure_agent_ui_built`). The deployer tars the entire `baal_agent` package over SSH, so anything in `src/baal_agent/webui/dist/` ships with the agent automatically — no deployer changes needed.

### Local UI environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_UI_ENABLED` | `True` | Whether to mount the SPA and CORS at all |
| `LOCAL_UI_CORS_ORIGINS` | `*` | Comma-separated list of allowed origins |
| `LOCAL_UI_DIST_PATH` | *(empty)* | Absolute path to the built SPA; empty falls back to `<package>/webui/dist` |

### Versioning

`AGENT_VERSION` in `baal_agent/__init__.py` is bumped whenever a feature requires server-side support — new endpoints, auth changes, API shape changes, etc. LiberClaw checks this value via `/health` to decide whether to offer features like direct chat.

**When bumping `AGENT_VERSION`, also update `MIN_DIRECT_CHAT_VERSION` in `src/liberclaw/routers/agents.py` to match** — otherwise LiberClaw will refuse to offer direct chat against agents at the new version.

Old agents (version < 4) do not have the `/info` endpoint, CORS, or the static SPA mount. The LiberClaw frontend handles this by showing an "Update your agent" modal instead of breaking.

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
| `OWNER_CHAT_ID` | — | Chat ID for subagent pending message delivery |
| `HEARTBEAT_INTERVAL` | `1800` | Seconds between heartbeat checks (0 to disable) |
| `MAX_TOOL_ITERATIONS` | `50` | Max tool calls per turn |
| `MAX_CONTEXT_TOKENS` | `0` | Context limit (0 = auto-detect from model) |
| `INFERENCE_TIMEOUT` | `300` | Seconds before inference timeout |

## License

AGPL-3.0
