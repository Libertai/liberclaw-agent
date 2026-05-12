"""File watcher definitions for scheduler-triggered agent tasks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileWatcher:
    id: str
    path: str
    task: str
    enabled: bool = True
    debounce: int = 60


def load_watchers(workspace_path: str) -> list[FileWatcher]:
    """Load watcher definitions from workspace/watchers.json."""
    path = Path(workspace_path) / "watchers.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load watchers.json: {e}")
        return []
    if not isinstance(data, list):
        logger.warning("watchers.json root is not a list; ignoring")
        return []

    watchers: list[FileWatcher] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        watcher_id = entry.get("id")
        watch_path = entry.get("path")
        task = entry.get("task")
        if not (watcher_id and watch_path and task):
            continue
        watchers.append(FileWatcher(
            id=str(watcher_id),
            path=str(watch_path),
            task=str(task),
            enabled=entry.get("enabled", True),
            debounce=max(0, int(entry.get("debounce", 60))),
        ))
    return watchers


def load_watcher_state(workspace_path: str) -> dict:
    """Load watcher state from workspace/.watcher_state.json."""
    path = Path(workspace_path) / ".watcher_state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load .watcher_state.json: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def save_watcher_state(workspace_path: str, state: dict) -> None:
    """Persist watcher state to workspace/.watcher_state.json."""
    path = Path(workspace_path) / ".watcher_state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def watched_fingerprint(workspace_path: str, watch_path: str) -> str | None:
    """Return a stable fingerprint for a watched file or directory."""
    workspace = Path(workspace_path).resolve()
    target = (workspace / watch_path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        logger.warning(f"Watcher path escapes workspace: {watch_path}")
        return None
    if not target.exists():
        return "missing"

    digest = hashlib.sha256()
    if target.is_file():
        stat = target.stat()
        digest.update(f"file:{target.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    if target.is_dir():
        for child in sorted(target.rglob("*")):
            if child.name in {".git", "__pycache__", "node_modules"}:
                continue
            try:
                rel = child.relative_to(workspace)
                stat = child.stat()
            except OSError:
                continue
            kind = "dir" if child.is_dir() else "file"
            digest.update(f"{kind}:{rel}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    return None


def watcher_due(
    watcher: FileWatcher,
    workspace_path: str,
    state: dict,
    *,
    now: float | None = None,
) -> bool:
    """Update state and return True when the watcher should trigger."""
    if not watcher.enabled:
        return False
    if now is None:
        now = time.time()

    fingerprint = watched_fingerprint(workspace_path, watcher.path)
    if fingerprint is None:
        return False

    entry = state.setdefault(watcher.id, {})
    previous = entry.get("fingerprint")
    entry["fingerprint"] = fingerprint
    entry["path"] = watcher.path
    entry["checked_at"] = now

    if previous is None:
        return False
    if previous == fingerprint:
        return False

    last_triggered = float(entry.get("last_triggered_at", 0))
    if watcher.debounce and (now - last_triggered) < watcher.debounce:
        return False

    entry["last_triggered_at"] = now
    return True
