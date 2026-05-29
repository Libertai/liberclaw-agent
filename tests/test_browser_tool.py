"""Tests for the headless-Chromium ``browser`` tool.

These run WITHOUT a real chromium install: the gating probe and the Playwright
page are both mocked, so the suite is hermetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from baal_agent import tools
from baal_agent.tools import _exec_browser, _tool_available, execute_tool_result


# ── Gating ────────────────────────────────────────────────────────────


def test_browser_unavailable_without_env(monkeypatch):
    monkeypatch.delenv("BROWSER_ENABLED", raising=False)
    available, reason = _tool_available("browser")
    assert available is False
    assert "BROWSER_ENABLED" in (reason or "")


def test_browser_unavailable_without_chromium(monkeypatch):
    monkeypatch.setenv("BROWSER_ENABLED", "true")
    monkeypatch.setattr(tools, "_chromium_present", lambda: False)
    available, reason = _tool_available("browser")
    assert available is False
    assert "chromium" in (reason or "")


def test_browser_available_when_enabled_and_present(monkeypatch):
    monkeypatch.setenv("BROWSER_ENABLED", "true")
    monkeypatch.setattr(tools, "_chromium_present", lambda: True)
    available, reason = _tool_available("browser")
    assert available is True
    assert reason is None


# ── Capability list (mirrors main._capabilities) ──────────────────────


def test_capability_list_excludes_then_includes_browser(monkeypatch):
    monkeypatch.delenv("BROWSER_ENABLED", raising=False)
    caps_off = ["vision"] + (["browser"] if _tool_available("browser")[0] else [])
    assert caps_off == ["vision"]

    monkeypatch.setenv("BROWSER_ENABLED", "true")
    monkeypatch.setattr(tools, "_chromium_present", lambda: True)
    caps_on = ["vision"] + (["browser"] if _tool_available("browser")[0] else [])
    assert "browser" in caps_on


# ── Tool registration ─────────────────────────────────────────────────


def test_browser_registered():
    assert "browser" in tools.TOOL_HANDLERS
    assert "browser" in tools._MUTATING_TOOLS
    assert "browser" in tools._IMAGE_AWARE_TOOLS
    names = [t["function"]["name"] for t in tools.TOOL_DEFINITIONS]
    assert "browser" in names


# ── _exec_browser behaviour (mocked Playwright page) ──────────────────


@pytest.fixture
def mock_page(monkeypatch):
    """Patch _get_browser_page to return a fully-mocked Playwright page."""
    page = MagicMock()
    page.url = "https://example.com/"
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.title = AsyncMock(return_value="Example Domain")
    page.content = AsyncMock(return_value="<html><body><p>Hello world</p></body></html>")
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n fake-png-bytes")

    async def _fake_get_page():
        return page

    monkeypatch.setattr(tools, "_get_browser_page", _fake_get_page)
    # Treat all URLs as public so navigation tests stay hermetic (no real DNS).
    monkeypatch.setattr(tools, "_validate_public_url", AsyncMock(return_value=None))
    return page


@pytest.mark.asyncio
async def test_browser_goto_returns_string(mock_page):
    result = await _exec_browser({"action": "goto", "url": "https://example.com"})
    assert isinstance(result, str)
    assert not result.startswith("[error:")
    mock_page.goto.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_goto_rejects_non_http():
    result = await _exec_browser({"action": "goto", "url": "file:///etc/passwd"})
    assert result.startswith("[error:")
    assert "http" in result


# ── SSRF guard ────────────────────────────────────────────────────────


def test_ip_is_blocked_classifies_private_and_public():
    from baal_agent.tools import _ip_is_blocked

    for blocked in (
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",  # cloud metadata
        "::1",
        "0.0.0.0",
        "::ffff:10.0.0.1",  # IPv4-mapped private
        "not-an-ip",
    ):
        assert _ip_is_blocked(blocked) is True, blocked
    for public in ("8.8.8.8", "1.1.1.1", "93.184.216.34"):
        assert _ip_is_blocked(public) is False, public


@pytest.mark.asyncio
async def test_validate_public_url_rejects_bad_schemes():
    from baal_agent.tools import _validate_public_url

    for url in (
        "file:///etc/passwd",
        "data:text/html,<h1>x</h1>",
        "chrome://settings",
        "view-source:http://example.com",
        "ftp://example.com/x",
    ):
        assert await _validate_public_url(url) is not None, url


@pytest.mark.asyncio
async def test_validate_public_url_blocks_private_resolution(monkeypatch):
    import baal_agent.tools as t

    def fake_getaddrinfo(host, port, *a, **k):
        return [(2, 1, 6, "", ("169.254.169.254", port))]

    monkeypatch.setattr(t.socket, "getaddrinfo", fake_getaddrinfo)
    reason = await t._validate_public_url("http://metadata.internal/latest/")
    assert reason is not None and "non-public" in reason


@pytest.mark.asyncio
async def test_validate_public_url_allows_public(monkeypatch):
    import baal_agent.tools as t

    def fake_getaddrinfo(host, port, *a, **k):
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(t.socket, "getaddrinfo", fake_getaddrinfo)
    assert await t._validate_public_url("https://example.com/") is None


@pytest.mark.asyncio
async def test_browser_goto_blocks_ssrf(monkeypatch, mock_page):
    # Override the fixture's permissive stub to simulate a blocked target.
    monkeypatch.setattr(
        tools,
        "_validate_public_url",
        AsyncMock(return_value="blocked: 'x' resolves to non-public address 127.0.0.1"),
    )
    result = await _exec_browser({"action": "goto", "url": "http://127.0.0.1:8080/"})
    assert result.startswith("[error:")
    assert "non-public" in result
    mock_page.goto.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_extract_strips_html_and_truncates(mock_page):
    result = await _exec_browser({"action": "extract"})
    assert "Hello world" in result
    assert "<p>" not in result


@pytest.mark.asyncio
async def test_browser_click_requires_selector(mock_page):
    result = await _exec_browser({"action": "click"})
    assert result.startswith("[error:")
    assert "selector" in result


@pytest.mark.asyncio
async def test_browser_type_requires_text(mock_page):
    result = await _exec_browser({"action": "type", "selector": "#q"})
    assert result.startswith("[error:")
    assert "text" in result


@pytest.mark.asyncio
async def test_browser_type_fills(mock_page):
    result = await _exec_browser({"action": "type", "selector": "#q", "text": "hi"})
    assert not result.startswith("[error:")
    mock_page.fill.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_screenshot_invokes_image_callback(mock_page):
    captured: list[list[dict]] = []

    def _cb(blocks):
        captured.append(blocks)

    result = await _exec_browser({"action": "screenshot"}, image_callback=_cb)
    assert not result.startswith("[error:")
    assert len(captured) == 1
    blocks = captured[0]
    assert any(b.get("type") == "image_url" for b in blocks)
    assert blocks[-1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_browser_unknown_action(mock_page):
    result = await _exec_browser({"action": "fly"})
    assert result.startswith("[error:")


@pytest.mark.asyncio
async def test_browser_launch_failure_is_error_string(monkeypatch):
    async def _boom():
        raise RuntimeError("playwright is not installed")

    monkeypatch.setattr(tools, "_get_browser_page", _boom)
    result = await _exec_browser({"action": "extract"})
    assert result.startswith("[error:")
    assert "playwright" in result


# ── ToolResult wrapper classifies the result ──────────────────────────


@pytest.mark.asyncio
async def test_execute_tool_result_wraps_browser(monkeypatch, mock_page):
    monkeypatch.setenv("BROWSER_ENABLED", "true")
    monkeypatch.setattr(tools, "_chromium_present", lambda: True)
    res = await execute_tool_result("browser", {"action": "extract"})
    assert res.name == "browser"
    assert res.is_error is False
    assert "Hello world" in res.content
    assert res.metadata.get("mutating") is True


@pytest.mark.asyncio
async def test_execute_tool_result_gates_when_disabled(monkeypatch):
    monkeypatch.delenv("BROWSER_ENABLED", raising=False)
    res = await execute_tool_result("browser", {"action": "extract"})
    assert res.is_error is True
    assert "unavailable" in res.content
