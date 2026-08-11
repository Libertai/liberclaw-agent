"""tools/list pagination bails out once the per-server tool cap is met,
rather than exhausting every page a server chooses to advertise."""

from __future__ import annotations

import pytest

from baal_agent.mcp_client import MCPClient
from baal_agent.mcp_transport import Transport

_PAGE_SIZE = 5


class FakeManyToolsTransport(Transport):
    """A server that never stops advertising a cursor: without an early bail
    this would force _MAX_TOOLS_LIST_PAGES (50) full round-trips."""

    def __init__(self):
        self._opened = False
        self.tools_list_calls = 0

    async def open(self) -> None:
        self._opened = True

    async def send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        if method == "initialize":
            return {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}
        if method == "tools/list":
            self.tools_list_calls += 1
            page = self.tools_list_calls
            tools = [
                {"name": f"t{page}_{i}", "description": "", "inputSchema": {}}
                for i in range(_PAGE_SIZE)
            ]
            return {"tools": tools, "nextCursor": str(page + 1)}
        raise AssertionError(f"unexpected method {method}")

    async def send_notification(self, method: str, params: dict) -> None:
        pass

    async def close(self) -> None:
        self._opened = False

    @property
    def connected(self) -> bool:
        return self._opened


@pytest.mark.asyncio
async def test_pagination_bails_early_once_cap_reached():
    client = MCPClient()
    transport = FakeManyToolsTransport()
    await client._connect_transport("probe", "http", transport)
    try:
        # 128-tool cap / 5 tools per page = 26 pages to reach it, well under
        # the 50-page backstop.
        assert transport.tools_list_calls == 26
        assert len(client.get_tool_definitions()) == 128
    finally:
        await client.disconnect_all()
