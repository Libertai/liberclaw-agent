"""Transport interface shared by the MCP transports.

Transports always raise ``MCPError``; they never return ``None``. Caller policy
lives with the caller — the handshake surfaces the message into health, while
tool calls degrade to a failure the circuit breaker counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MCPError(Exception):
    """Error from an MCP server or its transport."""


class Transport(ABC):
    @abstractmethod
    async def open(self) -> None:
        """Establish the connection. Raises MCPError on failure."""

    @abstractmethod
    async def send_request(self, method: str, params: dict, timeout: float) -> dict:
        """Send a JSON-RPC request and return its result. Raises MCPError."""

    @abstractmethod
    async def send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification. Best effort; never raises."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down. Safe to call on a transport that never opened."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the transport is currently usable."""

    @property
    def pending_count(self) -> int:
        """In-flight requests, reported in health."""
        return 0
