"""
Plugin Manager — discovery, loading, and lifecycle management.
"""
from __future__ import annotations
import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from .plugin_base import PluginBase, PluginManifest, PluginType


class PluginManager:
    """Discovers, loads, and manages all plugins."""

    def __init__(self, controller: Any = None):
        self.controller = controller
        self._plugins: dict[str, PluginBase] = {}
        self._plugin_dirs: list[Path] = [
            Path(__file__).parent / "builtin",
            Path.home() / ".medaxis" / "plugins",
        ]

    def add_plugin_dir(self, path: Path):
        self._plugin_dirs.append(path)

    def discover(self) -> list[PluginManifest]:
        """Discover all available plugins from plugin directories and entry points."""
        manifests = []

        # Discover from entry_points
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            entry_points = entry_points.select(group="medaxis.plugins")
        else:
            entry_points = entry_points.get("medaxis.plugins", [])
        for entry_point in entry_points:
            try:
                module = entry_point.load()
                if hasattr(module, "get_manifests"):
                    manifests.extend(module.get_manifests())
            except Exception as e:
                print(f"Error loading plugin {entry_point.name}: {e}")

        # Discover from plugin directories
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue
            for manifest_file in plugin_dir.rglob("manifest.json"):
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                        if isinstance(data.get("plugin_type"), str):
                            data["plugin_type"] = PluginType(data["plugin_type"])
                        manifests.append(PluginManifest(**data))
                except Exception as e:
                    print(f"Error loading manifest {manifest_file}: {e}")

        return manifests

    def load(self, manifest: PluginManifest) -> Optional[PluginBase]:
        """Load and activate a plugin from its manifest."""
        if manifest.name in self._plugins:
            return self._plugins[manifest.name]

        try:
            module_name, class_name = manifest.entry_point.rsplit(":", 1)
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)
            plugin = plugin_class()
            plugin.activate(self.controller)
            self._plugins[manifest.name] = plugin
            return plugin
        except Exception as e:
            print(f"Error loading plugin {manifest.name}: {e}")
            return None

    def load_all(self, manifests: list[PluginManifest] = None) -> int:
        """Load all discovered plugins."""
        if manifests is None:
            manifests = self.discover()
        count = 0
        for manifest in manifests:
            if self.load(manifest):
                count += 1
        return count

    def get(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> list[PluginBase]:
        return [p for p in self._plugins.values() if p.manifest().plugin_type == plugin_type]

    def deactivate_all(self):
        for plugin in self._plugins.values():
            try:
                plugin.deactivate()
            except Exception:
                pass
        self._plugins.clear()

    def unload(self, name: str) -> bool:
        """Deactivate and remove one plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        plugin.deactivate()
        return True

    def shutdown(self) -> None:
        self.deactivate_all()

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    @property
    def plugin_names(self) -> list[str]:
        return list(self._plugins.keys())
