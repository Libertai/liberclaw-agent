from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Settings for a deployed agent instance."""

    model_config = {"env_prefix": ""}

    agent_name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."
    model: str = "hermes-3-8b-tee"
    libertai_api_key: str
    agent_secret_hash: str  # SHA-256 hash of the shared secret
    port: int = 8080
    db_path: str = "agent.db"
    max_history: int = 100
    max_tool_iterations: int = 50
    workspace_path: str = "/opt/baal-agent/workspace"
    owner_chat_id: str = ""  # Telegram chat ID for heartbeat delivery
    heartbeat_interval: int = 1800  # 30 minutes
    max_context_tokens: int = 0  # 0 = auto-detect from model name
    generation_reserve: int = 4096  # tokens reserved for model output
    compaction_keep_messages: int = 20  # recent messages to preserve during compaction
    compaction_threshold: float = 0.75  # trigger compaction at this fraction of context budget
    inference_timeout: int = 180  # seconds — timeout for inference in the SSE loop
