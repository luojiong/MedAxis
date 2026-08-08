"""
Plugin Manager — discovery, loading, and lifecycle management.
"""
from __future__ import annotations
import importlib
import json
import logging
import inspect
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from .plugin_base import PluginBase, PluginManifest, PluginType

logger = logging.getLogger(__name__)


class _ToolOnlyPlugin(PluginBase):
    """Placeholder for plugins that only declare ``medaxis_tools``."""

    def __init__(self, manifest: PluginManifest) -> None:
        self._manifest = manifest

    def manifest(self) -> PluginManifest:
        return self._manifest

    def activate(self, controller) -> None:
        pass

    def deactivate(self) -> None:
        pass


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

        # Discover from filesystem plugin dirs (.py and compiled .pyd modules).
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.is_dir():
                continue
            for module_file in sorted(plugin_dir.glob("*.py")) + sorted(plugin_dir.glob("*.pyd")):
                manifest = self._manifest_from_module_file(module_file)
                if manifest is not None:
                    manifests.append(manifest)

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

    def _manifest_from_module_file(self, module_file: Path) -> Optional[PluginManifest]:
        """Build a manifest from a .py / .pyd module file on disk."""
        module_name = module_file.stem
        if module_file.suffix == ".py":
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                return None
        else:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                return None
        manifest_attr = getattr(module, "manifest", None)
        if isinstance(manifest_attr, dict):
            return PluginManifest(
                name=str(manifest_attr.get("name", module_name)),
                version=str(manifest_attr.get("version", "0.1.0")),
                entry_point=f"{module_name}:plugin",
                plugin_type=str(manifest_attr.get("type", "tool")),
                description=str(manifest_attr.get("description", "")),
            )
        return None

    def _register_native_tools(self, module, plugin_name: str) -> None:
        """Auto-expose a plugin's ``medaxis_tools`` list to the MCP router.

        Native (.pyd) plugins and python plugins may declare::

            medaxis_tools = [
                {"name": "...", "description": "...", "handler": callable,
                 "parameters_schema": {...}, "permission": "control"},
            ]
        """
        tools = getattr(module, "medaxis_tools", None)
        if not tools:
            return
        router = getattr(self.controller, "mcp_router", None)
        if router is None:
            return
        from mcp.auth import PermissionLevel
        from mcp.router import MCPTool

        for spec in tools:
            name = spec.get("name") or f"{plugin_name}_tool"
            handler = spec.get("handler")
            if handler is None:
                continue

            async def _dispatch(params, _h=handler):
                result = _h(params)
                if inspect.isawaitable(result):
                    return await result
                return result

            tool = MCPTool(
                name=name,
                description=spec.get("description", f"Plugin tool from {plugin_name}"),
                parameters_schema=spec.get("parameters_schema", {"type": "object"}),
                handler=_dispatch,
                permission=PermissionLevel(spec.get("permission", "control")),
            )
            router.register_tool(tool)
            logger.info("Exposed plugin tool %s to MCP", name)

    def load(self, manifest: PluginManifest) -> Optional[PluginBase]:
        """Load and activate a plugin from its manifest."""
        if manifest.name in self._plugins:
            return self._plugins[manifest.name]

        # File-based modules (.py / .pyd): import and auto-expose tools.
        module_name = manifest.entry_point.split(":", 1)[0]
        if module_name.endswith(".pyd"):
            module_name = module_name[:-4]
        if module_name:
            try:
                if module_name not in sys.modules:
                    for plugin_dir in self._plugin_dirs:
                        candidate = plugin_dir / (module_name + ".py")
                        if not candidate.is_file():
                            candidate = plugin_dir / (module_name + ".pyd")
                        if candidate.is_file():
                            if str(plugin_dir) not in sys.path:
                                sys.path.insert(0, str(plugin_dir))
                            break
                module = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning("Failed to import plugin module %s: %s", module_name, exc)
                return None
            self._register_native_tools(module, manifest.name)
            plugin_class = getattr(module, "plugin", None)
            if plugin_class is None:
                # Tool-only plugin (no PluginBase subclass): mark as loaded
                # with a lightweight placeholder.
                self._plugins[manifest.name] = _ToolOnlyPlugin(manifest)
                return self._plugins[manifest.name]
            plugin = plugin_class()
            plugin.activate(self.controller)
            self._plugins[manifest.name] = plugin
            return plugin

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
