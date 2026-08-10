"""MCP server config decoding.

The deployer writes MCP_SERVERS_B64 because systemd's EnvironmentFile parser
drops the backslash on bare-value escapes (\\u00e9 becomes u00e9) and would
mutate raw JSON.
"""

from __future__ import annotations

import base64
import json

import pytest

from baal_agent.tools import resolve_mcp_servers_json


class _Settings:
    def __init__(self, mcp_servers="", mcp_servers_b64=""):
        self.mcp_servers = mcp_servers
        self.mcp_servers_b64 = mcp_servers_b64


SERVERS = [{"name": "notes", "transport": "stdio", "command": "npx",
            "env": {"NOTE": "café", "Q": 'a"b\\c'}}]


def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


def _b64_text(text):
    return base64.b64encode(text.encode()).decode()


def test_prefers_b64_and_round_trips_exactly():
    settings = _Settings(mcp_servers_b64=_b64(SERVERS))
    assert json.loads(resolve_mcp_servers_json(settings)) == SERVERS


def test_falls_back_to_plain_when_b64_absent():
    raw = json.dumps(SERVERS)
    assert resolve_mcp_servers_json(_Settings(mcp_servers=raw)) == raw


def test_b64_wins_when_both_present():
    settings = _Settings(mcp_servers='[{"name":"stale"}]', mcp_servers_b64=_b64(SERVERS))
    assert json.loads(resolve_mcp_servers_json(settings))[0]["name"] == "notes"


def test_empty_when_neither_set():
    assert resolve_mcp_servers_json(_Settings()) == ""


def test_undecodable_b64_falls_back_rather_than_crashing():
    settings = _Settings(mcp_servers='[{"name":"plain"}]', mcp_servers_b64="!!!not-base64!!!")
    assert json.loads(resolve_mcp_servers_json(settings))[0]["name"] == "plain"


def test_system_prompt_b64_is_decoded_into_settings(monkeypatch):
    """A multi-paragraph prompt must arrive with real newlines."""
    from baal_agent.config import AgentSettings

    prompt = "Line one.\nLine two.\n\nFinal paragraph."
    monkeypatch.setenv("LIBERTAI_API_KEY", "k")
    monkeypatch.setenv("AGENT_SECRET_HASH", "h")
    monkeypatch.setenv("SYSTEM_PROMPT", "ignored")
    monkeypatch.setenv("SYSTEM_PROMPT_B64", _b64_text(prompt))
    assert AgentSettings().system_prompt == prompt


@pytest.mark.parametrize("absent", ["unset", "empty"])
def test_plain_system_prompt_used_when_b64_absent(monkeypatch, absent):
    from baal_agent.config import AgentSettings

    monkeypatch.setenv("LIBERTAI_API_KEY", "k")
    monkeypatch.setenv("AGENT_SECRET_HASH", "h")
    monkeypatch.setenv("SYSTEM_PROMPT", "be helpful")
    if absent == "unset":
        monkeypatch.delenv("SYSTEM_PROMPT_B64", raising=False)
    else:
        monkeypatch.setenv("SYSTEM_PROMPT_B64", "")
    assert AgentSettings().system_prompt == "be helpful"


def test_undecodable_system_prompt_b64_keeps_plain(monkeypatch, caplog):
    import logging

    from baal_agent.config import AgentSettings

    monkeypatch.setenv("LIBERTAI_API_KEY", "k")
    monkeypatch.setenv("AGENT_SECRET_HASH", "h")
    monkeypatch.setenv("SYSTEM_PROMPT", "fallback text")
    monkeypatch.setenv("SYSTEM_PROMPT_B64", "!!!not-base64!!!")
    with caplog.at_level(logging.WARNING):
        settings = AgentSettings()
    assert settings.system_prompt == "fallback text"
    assert "not valid base64" in caplog.text
