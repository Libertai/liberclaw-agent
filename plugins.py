"""Drop-in plugin system for baal-agent.

Plugins are Python files in ``workspace/plugins/``. Each plugin module can
define hook functions that are called at various points in the agent lifecycle:

- ``on_session_start(chat_id: str) -> None``
- ``on_session_end(chat_id: str) -> None``
- ``pre_inference(messages: list[dict]) -> list[dict]``
- ``post_inference(response_text: str) -> str``
- ``pre_tool(arguments: dict, tool_name: str) -> dict``
- ``post_tool(result: str, tool_name: str) -> str``

A broken plugin never crashes the agent -- all hook calls are wrapped in
try/except and failures are logged as warnings.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
from types import ModuleType

logger = logging.getLogger(__name__)


class PluginManager:
    """Load and manage workspace plugins."""

    def __init__(self, workspace_path: str):
        self.plugins_dir = os.path.join(workspace_path, "plugins")
        self.plugins: list[ModuleType] = []

    def load_plugins(self) -> None:
        """Scan workspace/plugins/*.py and import each module.

        Previously loaded plugins are discarded so that reloading picks up
        new or modified files.
        """
        self.plugins.clear()

        if not os.path.isdir(self.plugins_dir):
            return

        for filename in sorted(os.listdir(self.plugins_dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            filepath = os.path.join(self.plugins_dir, filename)
            module_name = f"baal_plugin_{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    logger.warning(f"Plugin {filename}: could not create module spec")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.plugins.append(module)
                logger.info(f"Loaded plugin: {filename}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {filename}: {e}")

    async def fire(self, hook_name: str, *args, **kwargs) -> None:
        """Call a notification hook on all plugins that define it.

        Used for hooks with no return value (on_session_start, on_session_end).
        Exceptions are logged but never propagated.
        """
        for plugin in self.plugins:
            fn = getattr(plugin, hook_name, None)
            if fn is None:
                continue
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                name = getattr(plugin, "__name__", "unknown")
                logger.warning(f"Plugin {name}.{hook_name}() failed: {e}")

    async def fire_modify(self, hook_name: str, value, *args, **kwargs):
        """Chain a transformation hook through all plugins.

        Each plugin receives the previous plugin's return value as the first
        positional argument. Used for pre_tool, post_tool, pre_inference,
        post_inference.
        """
        for plugin in self.plugins:
            fn = getattr(plugin, hook_name, None)
            if fn is None:
                continue
            try:
                result = fn(value, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    value = result
            except Exception as e:
                name = getattr(plugin, "__name__", "unknown")
                logger.warning(f"Plugin {name}.{hook_name}() failed: {e}")
        return value
