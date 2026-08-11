"""Registry of MCP servers and their namespaced tools."""

from __future__ import annotations


class ToolRegistry:
    def __init__(self):
        self._servers: dict[str, object] = {}
        self._tools: dict[str, object] = {}

    @property
    def servers(self) -> dict:
        return self._servers

    def register(self, conn, tools: dict) -> None:
        prior = self._servers.get(conn.name)
        if prior is not None:
            # A same-named conn already held a slot (reconnect). Drop only
            # the tools it no longer offers — a key kept by both stays in
            # its slot, since dict.update() below won't move it.
            for key in list(getattr(prior, "tools", {})):
                if key not in tools:
                    self._tools.pop(key, None)
        self._servers[conn.name] = conn
        conn.tools = dict(tools)  # replace_tools and remove both read this back
        self._tools.update(tools)

    def replace_tools(self, conn, tools: dict) -> bool:
        """Swap one server's tools. Returns False if that server changed.

        The caller computed ``tools`` across a network await, so the connection
        may have been replaced meanwhile. Identity — not name — decides, and
        from the check to the last mutation there is no await point, which is
        what makes this atomic in single-threaded asyncio.

        Rebuilds in place so each server keeps its slot: ``get_tool_definitions``
        iterates insertion order, and that order feeds the static prompt prefix.
        """
        if self._servers.get(conn.name) is not conn:
            return False
        rebuilt = {}
        replaced = False
        for key, value in self._tools.items():
            if key in conn.tools:
                if not replaced:
                    rebuilt.update(tools)
                    replaced = True
                continue
            rebuilt[key] = value
        if not replaced:
            rebuilt.update(tools)
        self._tools = rebuilt
        conn.tools = dict(tools)
        return True

    def remove(self, name: str) -> None:
        conn = self._servers.pop(name, None)
        if conn is None:
            return
        for key in list(getattr(conn, "tools", {})):
            self._tools.pop(key, None)

    def get(self, namespaced: str):
        return self._tools.get(namespaced)

    def all_tools(self) -> dict:
        return self._tools
