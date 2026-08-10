"""MCP server config decoding.

The deployer writes MCP_SERVERS_B64 because systemd's EnvironmentFile parser
decodes backslash escapes in bare values and would mutate raw JSON.
"""

from __future__ import annotations

import base64
import json

from baal_agent.tools import resolve_mcp_servers_json


class _Settings:
    def __init__(self, mcp_servers="", mcp_servers_b64=""):
        self.mcp_servers = mcp_servers
        self.mcp_servers_b64 = mcp_servers_b64


SERVERS = [{"name": "notes", "transport": "stdio", "command": "npx",
            "env": {"NOTE": "café", "Q": 'a"b\\c'}}]


def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


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
