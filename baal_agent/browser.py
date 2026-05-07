"""Browser automation module — Playwright-based headless browser control.

Provides a stateful browser instance that persists across tool calls within
the same conversation turn, enabling multi-step web interactions (navigate,
click, type, screenshot, extract content).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from baal_agent.image_utils import encode_bytes_to_data_uri

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_PLAYWRIGHT_TIMEOUT = 30_000  # 30 seconds in ms
_PLAYWRIGHT_NAVIGATE_TIMEOUT = 60_000  # 60 seconds for navigation
_MAX_SCREENSHOT_WIDTH = 1280
_MAX_SCREENSHOT_HEIGHT = 720
_MAX_PAGE_CONTENT = 30_000  # max chars for page text content

# ── Lazy-init browser ─────────────────────────────────────────────────

_browser_lock = asyncio.Lock()
_playwright = None
_browser = None
_context = None
_page = None


async def _ensure_browser() -> None:
    """Lazily initialize Playwright browser instance."""
    global _playwright, _browser, _context, _page

    async with _browser_lock:
        if _page is not None:
            # Check page is still alive
            try:
                await _page.evaluate("1")
                return
            except Exception:
                pass

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && "
                "playwright install chromium"
            )

        if _playwright is None:
            _playwright = await async_playwright().start()

        headless = os.environ.get("BROWSER_HEADLESS", "true").lower() in ("true", "1", "yes")
        _browser = await _playwright.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
            ],
        )

        _context = await _browser.new_context(
            viewport={"width": _MAX_SCREENSHOT_WIDTH, "height": _MAX_SCREENSHOT_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
            ignore_https_errors=True,
        )

        _page = await _context.new_page()
        _page.set_default_timeout(_PLAYWRIGHT_TIMEOUT)
        logger.info("Browser initialized (headless=%s)", headless)


async def _ensure_page_closed() -> None:
    """Close the current page (not the browser) so we start fresh next call."""
    global _page
    if _page is not None:
        try:
            await _page.close()
        except Exception:
            pass
        _page = None


async def shutdown_browser() -> None:
    """Cleanup browser resources. Called during application shutdown."""
    global _playwright, _browser, _context, _page
    try:
        if _page is not None:
            await _page.close()
    except Exception:
        pass
    _page = None
    try:
        if _context is not None:
            await _context.close()
    except Exception:
        pass
    _context = None
    try:
        if _browser is not None:
            await _browser.close()
    except Exception:
        pass
    _browser = None
    try:
        if _playwright is not None:
            await _playwright.stop()
    except Exception:
        pass
    _playwright = None


# ── Tool actions ──────────────────────────────────────────────────────


async def _navigate(url: str) -> str:
    """Navigate to a URL and return page info."""
    await _ensure_browser()
    try:
        response = await _page.goto(url, timeout=_PLAYWRIGHT_NAVIGATE_TIMEOUT, wait_until="domcontentloaded")
        # Wait a bit for JS rendering
        await asyncio.sleep(1.5)
        title = await _page.title()
        status = response.status if response else "unknown"
        # Extract visible text content
        text = await _page.evaluate(
            "() => document.body?.innerText || ''"
        )
        text = text.strip()
        truncated = False
        if len(text) > _MAX_PAGE_CONTENT:
            text = text[:_MAX_PAGE_CONTENT]
            truncated = True

        link_count = await _page.evaluate("() => document.querySelectorAll('a[href]').length")
        input_count = await _page.evaluate("() => document.querySelectorAll('input, textarea, select, button').length")

        result = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Status: {status}\n"
            f"Page elements: {link_count} links, {input_count} interactive elements\n\n"
        )
        if text:
            result += f"Page content:\n{text}"
            if truncated:
                result += "\n\n... (content truncated)"
        else:
            result += "(no visible text content — page may be image-heavy or JS-rendered)"

        return result
    except Exception as e:
        return f"[browser error navigating to {url}: {e}]"


async def _click(selector: str) -> str:
    """Click an element identified by CSS selector."""
    await _ensure_browser()
    try:
        await _page.wait_for_selector(selector, state="visible", timeout=_PLAYWRIGHT_TIMEOUT)
        await _page.click(selector)
        await asyncio.sleep(1)
        # Return updated page state
        title = await _page.title()
        text = await _page.evaluate("() => document.body?.innerText || ''")
        text = text.strip()
        if len(text) > _MAX_PAGE_CONTENT:
            text = text[:_MAX_PAGE_CONTENT]
        return (
            f"Clicked: {selector}\n"
            f"Title: {title}\n"
            f"Page content:\n{text}"
        )
    except Exception as e:
        return f"[browser error clicking '{selector}': {e}]"


async def _type_text(selector: str, text: str) -> str:
    """Type text into an input field identified by CSS selector."""
    await _ensure_browser()
    try:
        await _page.wait_for_selector(selector, state="visible", timeout=_PLAYWRIGHT_TIMEOUT)
        await _page.fill(selector, text)
        return f"Typed into {selector}: {text[:200]}{'...' if len(text) > 200 else ''}"
    except Exception as e:
        return f"[browser error typing into '{selector}': {e}]"


async def _press_key(key: str) -> str:
    """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""
    await _ensure_browser()
    try:
        await _page.keyboard.press(key)
        await asyncio.sleep(0.5)
        title = await _page.title()
        return f"Pressed key: {key}\nTitle: {title}"
    except Exception as e:
        return f"[browser error pressing key '{key}': {e}]"


async def _screenshot(full_page: bool = False) -> str:
    """Take a screenshot of the current page. Returns a data URI for visual analysis."""
    await _ensure_browser()
    try:
        screenshot_opts = {"type": "png", "full_page": full_page}
        png_bytes = await _page.screenshot(**screenshot_opts)
        data_uri = encode_bytes_to_data_uri(png_bytes, "image/png")
        # Save to workspace/tmp for file delivery
        if _workspace_path:
            tmp_dir = Path(_workspace_path) / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                task_name = asyncio.current_task().get_name()
            except (RuntimeError, AttributeError):
                task_name = "browser"
            fname = f"screenshot_{task_name}.png"
            fname = re.sub(r"[^a-zA-Z0-9._-]", "_", fname)
            fpath = tmp_dir / fname
            fpath.write_bytes(png_bytes)
            return (
                f"[Screenshot saved to tmp/{fname} ({len(png_bytes):,} bytes)]\n"
                f"[Screenshot available for visual analysis]"
            )
        return f"[Screenshot captured ({len(png_bytes):,} bytes)]"
    except Exception as e:
        return f"[browser error taking screenshot: {e}]"


async def _get_html() -> str:
    """Get the full HTML source of the current page."""
    await _ensure_browser()
    try:
        html = await _page.content()
        if len(html) > _MAX_PAGE_CONTENT:
            html = html[:_MAX_PAGE_CONTENT] + f"\n\n... truncated ({len(html)} total chars)"
        return html
    except Exception as e:
        return f"[browser error getting HTML: {e}]"


async def _evaluate(expression: str) -> str:
    """Evaluate JavaScript in the page context."""
    await _ensure_browser()
    try:
        result = await _page.evaluate(expression)
        if result is None:
            return "(no return value)"
        text = str(result)
        if len(text) > _MAX_PAGE_CONTENT:
            text = text[:_MAX_PAGE_CONTENT] + f"\n\n... truncated ({len(text)} total chars)"
        return text
    except Exception as e:
        return f"[browser JS eval error: {e}]"


async def _close_page() -> str:
    """Close the current page to reset browser state."""
    await _ensure_page_closed()
    return "Page closed. Browser ready for fresh navigation."


# ── Workspace path ────────────────────────────────────────────────────

_workspace_path: str | None = None


def configure_browser(workspace_path: str | None) -> None:
    """Set workspace path for screenshot storage."""
    global _workspace_path
    _workspace_path = workspace_path


# ── Tool executor ─────────────────────────────────────────────────────


async def _exec_browser(args: dict) -> str:
    """Main browser tool executor — dispatches to action-specific functions.

    Actions: navigate, click, type, press, screenshot, html, evaluate, close
    """
    action = args.get("action", "")
    if not action:
        return "[error: missing required 'action' parameter]"

    if action == "navigate":
        url = args.get("url")
        if not url:
            return "[error: 'url' is required for navigate]"
        # Validate URL scheme to prevent file:// and other unsafe schemes
        url_lower = url.strip().lower()
        if not url_lower.startswith(("http://", "https://")):
            return "[error: only http:// and https:// URLs are allowed.]"
        return await _navigate(url)

    elif action == "click":
        selector = args.get("selector")
        if not selector:
            return "[error: 'selector' is required for click]"
        return await _click(selector)

    elif action == "type":
        selector = args.get("selector")
        text = args.get("text", "")
        if not selector:
            return "[error: 'selector' is required for type]"
        return await _type_text(selector, text)

    elif action == "press":
        key = args.get("key")
        if not key:
            return "[error: 'key' is required for press]"
        return await _press_key(key)

    elif action == "screenshot":
        full_page = args.get("full_page", False)
        return await _screenshot(full_page=full_page)

    elif action == "html":
        return await _get_html()

    elif action == "evaluate":
        expression = args.get("expression")
        if not expression:
            return "[error: 'expression' is required for evaluate]"
        return await _evaluate(expression)

    elif action == "close":
        return await _close_page()

    else:
        return (
            f"[error: unknown action '{action}'. "
            f"Valid actions: navigate, click, type, press, screenshot, html, evaluate, close]"
        )


# ── Tool definition ───────────────────────────────────────────────────

BROWSER_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "browser",
        "description": (
            "Control a headless web browser. Can navigate to URLs, click elements, "
            "type text, press keys, take screenshots, get HTML source, evaluate JavaScript, "
            "or close the current page. The browser state persists across calls within a "
            "session — navigate to a page, then interact with it step by step. "
            "Use 'screenshot' when you need to see the page visually (rendered JS, dynamic content). "
            "Use evaluate with JavaScript expressions like "
            "'document.title' or 'document.querySelectorAll(\"a\").length'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "click", "type", "press", "screenshot", "html", "evaluate", "close"],
                    "description": "Browser action to perform.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (required for navigate).",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element (required for click, type).",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the element (for type action).",
                },
                "key": {
                    "type": "string",
                    "description": "Keyboard key to press: Enter, Tab, Escape, ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Backspace, Delete, etc. (for press action).",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "If true, take a full-page screenshot (for screenshot action). Default: false (viewport only).",
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context (for evaluate action).",
                },
            },
            "required": ["action"],
        },
    },
}
