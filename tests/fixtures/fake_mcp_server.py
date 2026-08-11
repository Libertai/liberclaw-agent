"""A minimal MCP server over stdio, for tests.

Run with `python -u` — block-buffered stdout on a pipe would make the client's
readline() hang until its timeout instead of returning promptly.
"""

import json
import sys

TOOLS = [
    {"name": "alpha", "description": "first", "inputSchema": {"type": "object"}},
    {"name": "beta", "description": "second", "inputSchema": {"type": "object"}},
]


def reply(msg_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, msg_id = msg.get("method"), msg.get("id")

        if method == "initialize":
            reply(msg_id, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            })
        elif method == "notifications/initialized":
            continue  # notification: no reply
        elif method == "tools/list":
            cursor = (msg.get("params") or {}).get("cursor")
            if cursor is None:
                reply(msg_id, {"tools": [TOOLS[0]], "nextCursor": "page2"})
            else:
                reply(msg_id, {"tools": [TOOLS[1]]})
        elif method == "tools/call":
            name = msg["params"]["name"]
            reply(msg_id, {"content": [{"type": "text", "text": f"called {name}"}]})
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }) + "\n")
            sys.stdout.flush()


main()
