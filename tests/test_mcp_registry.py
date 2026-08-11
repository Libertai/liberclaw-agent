"""ToolRegistry atomicity and ordering.

replace_tools runs after a network await, so the server it was called for may
have been swapped underneath it. It re-checks identity and mutates without
suspending, and preserves each server's slot so the tool ordering — and the
prompt prefix built from it — stays stable.
"""

from __future__ import annotations

from baal_agent.mcp_registry import ToolRegistry


class FakeConn:
    def __init__(self, name):
        self.name = name


def info(name):
    return {"name": name}


def test_register_then_get():
    reg = ToolRegistry()
    conn = FakeConn("a")
    reg.register(conn, {"mcp_a_one": info("one")})
    assert reg.get("mcp_a_one") == info("one")


def test_replace_swaps_tools_for_that_server_only():
    reg = ToolRegistry()
    a, b = FakeConn("a"), FakeConn("b")
    reg.register(a, {"mcp_a_one": info("one")})
    reg.register(b, {"mcp_b_x": info("x")})
    assert reg.replace_tools(a, {"mcp_a_two": info("two")}) is True
    assert reg.get("mcp_a_one") is None
    assert reg.get("mcp_a_two") == info("two")
    assert reg.get("mcp_b_x") == info("x")


def test_replace_bails_when_server_was_swapped():
    reg = ToolRegistry()
    original = FakeConn("a")
    reg.register(original, {"mcp_a_one": info("one")})
    replacement = FakeConn("a")
    reg.register(replacement, {"mcp_a_new": info("new")})
    assert reg.get("mcp_a_one") is None, "the replaced conn's tools were left behind"
    # `original` is no longer the registered conn for "a".
    assert reg.replace_tools(original, {"mcp_a_stale": info("stale")}) is False
    assert reg.get("mcp_a_stale") is None
    assert reg.get("mcp_a_new") == info("new")


def test_replace_updates_values_when_names_are_unchanged():
    """Same names, changed metadata: the new values must win.

    A periodic refresh usually returns an identical name set, so a comparison
    that only looked at keys would discard every metadata update.
    """
    reg = ToolRegistry()
    a = FakeConn("a")
    reg.register(a, {"mcp_a_1": info("old")})
    assert reg.replace_tools(a, {"mcp_a_1": info("new")}) is True
    assert reg.get("mcp_a_1") == info("new")


def test_reregister_preserves_ordering_for_unchanged_tools():
    """A reconnect must not move a server's tools to the end.

    Insertion order feeds the static prompt prefix, so re-ordering an
    unchanged tool set would re-tokenize it for no reason.
    """
    reg = ToolRegistry()
    a, b, c = FakeConn("a"), FakeConn("b"), FakeConn("c")
    reg.register(a, {"mcp_a_1": info("1")})
    reg.register(b, {"mcp_b_1": info("1")})
    reg.register(c, {"mcp_c_1": info("1")})
    reconnected = FakeConn("a")
    reg.register(reconnected, {"mcp_a_1": info("1")})
    assert list(reg.all_tools()) == ["mcp_a_1", "mcp_b_1", "mcp_c_1"]


def test_replace_preserves_ordering():
    reg = ToolRegistry()
    a, b, c = FakeConn("a"), FakeConn("b"), FakeConn("c")
    reg.register(a, {"mcp_a_1": info("1")})
    reg.register(b, {"mcp_b_1": info("1")})
    reg.register(c, {"mcp_c_1": info("1")})
    reg.replace_tools(b, {"mcp_b_2": info("2")})
    assert list(reg.all_tools()) == ["mcp_a_1", "mcp_b_2", "mcp_c_1"]


def test_replace_is_a_noop_when_unchanged():
    reg = ToolRegistry()
    a = FakeConn("a")
    tools = {"mcp_a_1": info("1")}
    reg.register(a, tools)
    before = list(reg.all_tools())
    assert reg.replace_tools(a, {"mcp_a_1": info("1")}) is True
    assert list(reg.all_tools()) == before


def test_remove_drops_server_and_its_tools():
    reg = ToolRegistry()
    a = FakeConn("a")
    reg.register(a, {"mcp_a_1": info("1")})
    reg.remove("a")
    assert reg.get("mcp_a_1") is None
    assert "a" not in reg.servers
