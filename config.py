from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Settings for a deployed agent instance."""

    model_config = {"env_prefix": ""}

    agent_name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."
    model: str = "claw-large"
    vision_delegation_model: str = (
        "claw-flash"  # model used to describe images for non-vision agents
    )
    libertai_api_key: str
    agent_secret_hash: str  # SHA-256 hash of the shared secret
    port: int = 8080
    db_path: str = "agent.db"
    max_history: int = 100
    max_tool_iterations: int = 150
    workspace_path: str = "/opt/baal-agent/workspace"
    owner_chat_id: str = ""  # Telegram chat ID for subagent pending message delivery
    heartbeat_interval: int = 1800  # seconds (0 = disabled)
    max_context_tokens: int = 0  # 0 = auto-detect from model name
    max_images_per_request: int = (
        4  # cap images sent to inference (0 = no cap); older ones dropped
    )
    generation_reserve: int = 4096  # tokens reserved for model output
    compaction_keep_messages: int = (
        20  # max recent messages to preserve during compaction
    )
    compaction_keep_min: int = (
        6  # minimum messages to keep even under extreme context pressure
    )
    compaction_threshold: float = (
        0.75  # trigger compaction at this fraction of context budget
    )
    compaction_flush_enabled: bool = True  # run memory flush before compaction
    auto_skill_threshold: int = 10  # tool calls to trigger skill nudge (0 = disabled)
    inference_timeout: int = 300  # seconds — timeout for inference in the SSE loop
    telegram_bot_token: str = ""  # Empty = Telegram disabled
    owner_telegram_id: str = ""  # Auto-allowed in contact list
    mcp_servers: str = ""  # JSON: [{"name": "...", "transport": "stdio", "command": "...", "args": [...], "env": {...}}]
    local_ui_enabled: bool = True
    local_ui_cors_origins: str = "*"  # comma-separated
    local_ui_dist_path: str = ""  # empty = <package>/webui/dist
    agent_fqdn: str = (
        ""  # set by the deployer, used in the system prompt; empty = standalone/local
    )
    tool_policy: str = "full-auto"  # full-auto, auto-read, ask-before-write, ask-before-shell, locked-down
    tool_allowlist: str = (
        ""  # comma-separated tool names; empty = no explicit allowlist
    )
    tool_denylist: str = (
        ""  # comma-separated tool names denied even if otherwise allowed
    )
    browser_enabled: bool = (
        False  # gate the browser tool (paid agents only; chromium installed at deploy)
    )
    runtime_event_retention_days: int = (
        30  # rolling window for runtime_events; 0 disables pruning
    )
