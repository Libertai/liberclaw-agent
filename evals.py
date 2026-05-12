"""Deterministic eval harness for the local agent runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import baal_agent.tools as tools_module

# Concurrent eval runs would race on the module-globals they monkey-patch
# (inference + db). Serialize them so a parent test runner or a misbehaving
# caller can't contaminate a live agent session.
_EVAL_LOCK = asyncio.Lock()


@dataclass
class EvalResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    final: str | None = None


class ScriptedInference:
    """Inference stub that returns JSON-scripted responses in order."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("ScriptedInference exhausted responses")
        response = self.responses.pop(0)
        tool_calls = [
            SimpleNamespace(
                id=call.get("id", f"call_{index + 1}"),
                function=SimpleNamespace(
                    name=call["name"],
                    arguments=json.dumps(call.get("arguments", {})),
                ),
            )
            for index, call in enumerate(response.get("tool_calls") or [])
        ]
        return SimpleNamespace(
            content=response.get("content", ""),
            tool_calls=tool_calls or None,
        )


class InMemoryEvalDatabase:
    """Minimal database adapter for standalone deterministic eval runs."""

    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content,
        tool_calls=None,
        tool_call_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.messages.append({
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "metadata": metadata or {},
        })

    async def get_history(self, chat_id: str, limit: int | None = None) -> list[dict]:
        messages = [
            message for message in self.messages
            if message["chat_id"] == chat_id
        ]
        return messages[-limit:] if limit else messages

    async def add_event(self, chat_id: str, event_type: str, payload: dict) -> None:
        self.events.append({
            "id": len(self.events) + 1,
            "chat_id": chat_id,
            "type": event_type,
            "payload": payload,
        })

    async def get_events(
        self,
        chat_id: str,
        *,
        limit: int = 200,
        after_id: int | None = None,
    ) -> list[dict]:
        events = [
            event for event in self.events
            if event["chat_id"] == chat_id
        ]
        if after_id is not None:
            events = [event for event in events if event["id"] > after_id]
        return events[-limit:]


def load_eval_case(path: str | Path) -> dict[str, Any]:
    """Load one eval case from JSON."""
    return json.loads(Path(path).read_text())


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load eval cases from a JSON file or directory of JSON files."""
    eval_path = Path(path)
    if eval_path.is_dir():
        return [
            load_eval_case(child)
            for child in sorted(eval_path.glob("*.json"))
        ]
    return [load_eval_case(eval_path)]


def _resolve_eval_workspace_path(workspace: Path, rel_path: str) -> Path:
    """Reject workspace-escaping paths in case files.

    The eval JSON is dev-supplied today, but anyone running CI on
    contributed eval cases needs this defense — `{"../../../etc/foo": ...}`
    would otherwise write outside the harness.
    """
    raw = rel_path.strip()
    if not raw or raw.startswith("/") or Path(raw).is_absolute():
        raise ValueError(f"eval workspace_files path is absolute: {rel_path!r}")
    resolved = (workspace / raw).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(
            f"eval workspace_files path escapes workspace: {rel_path!r}"
        ) from exc
    return resolved


async def run_agent_eval_case(
    main_module,
    case: dict[str, Any],
    *,
    workspace_path: str | Path | None = None,
    db: Any | None = None,
) -> EvalResult:
    """Run one scripted eval case through baal_agent.main._run_agent_turn.

    Caller may pass ``db`` to override ``main_module.db`` for the duration
    of the run (the InMemoryEvalDatabase is a natural choice). Without an
    override, the run shares the live agent DB, which contaminates real
    history and is rarely what tests want.
    """
    name = case.get("name", "unnamed")
    workspace = Path(workspace_path or main_module.settings.workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    for rel_path, content in case.get("workspace_files", {}).items():
        target = _resolve_eval_workspace_path(workspace, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    scripted = ScriptedInference(case.get("responses", []))
    chat_id = case.get("chat_id", f"eval:{name}")

    # Serialize the monkey-patch window so two concurrent eval runs can't
    # observe each other's swapped-in inference / db.
    async with _EVAL_LOCK:
        original_inference = main_module.inference
        original_db = main_module.db if db is not None else None
        original_tool_state = (
            tools_module._workspace_path,
            tools_module._db,
            tools_module._inference,
            tools_module._model,
        )
        main_module.inference = scripted
        if db is not None:
            main_module.db = db
            tools_module.configure_tools(
                str(workspace),
                db=db,
                inference=scripted,
                model=getattr(main_module.settings, "model", ""),
            )
        try:
            final = await main_module._run_agent_turn(
                case["input"],
                chat_id=chat_id,
                max_iterations=case.get("max_iterations", 5),
                platform=case.get("platform", "eval"),
            )
        finally:
            main_module.inference = original_inference
            if db is not None:
                main_module.db = original_db
                (
                    tools_module._workspace_path,
                    tools_module._db,
                    tools_module._inference,
                    tools_module._model,
                ) = original_tool_state

    failures: list[str] = []
    expect = case.get("expect", {})
    expected_final = expect.get("final_contains")
    if expected_final and expected_final not in (final or ""):
        failures.append(f"final did not contain {expected_final!r}")

    for rel_path, expected_content in expect.get("files", {}).items():
        target = _resolve_eval_workspace_path(workspace, rel_path)
        if not target.exists():
            failures.append(f"expected file missing: {rel_path}")
            continue
        actual = target.read_text()
        if expected_content not in actual:
            failures.append(f"file {rel_path} did not contain expected text")

    eval_db = db if db is not None else main_module.db
    if "history_roles" in expect:
        history = await eval_db.get_history(chat_id)
        roles = [message["role"] for message in history]
        if roles != expect["history_roles"]:
            failures.append(f"history roles {roles!r} != {expect['history_roles']!r}")

    if "events" in expect:
        events = await eval_db.get_events(chat_id)
        event_types = [event["type"] for event in events]
        for event_type in expect["events"]:
            if event_type not in event_types:
                failures.append(f"event {event_type!r} was not emitted")

    if scripted.responses:
        failures.append(f"{len(scripted.responses)} scripted responses were unused")

    return EvalResult(
        name=name,
        passed=not failures,
        failures=failures,
        final=final,
    )
