"""
Plugin system — Algorithm, Model, Tool, View, and Reader plugin types.
"""
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from enum import Enum


class PluginType(Enum):
    ALGORITHM = "algorithm"
    MODEL = "model"           # AI model (API endpoint)
    TOOL = "tool"             # Interactive tool (measurement, annotation)
    VIEW = "view"             # New view type
    READER = "reader"         # New file format reader


@dataclass
class PluginManifest:
    """Plugin metadata for discovery."""
    name: str
    version: str
    plugin_type: PluginType
    author: str = ""
    description: str = ""
    entry_point: str = ""     # module:class
    dependencies: list[str] = field(default_factory=list)


class PluginBase(ABC):
    """Base class for all plugins."""

    @abstractmethod
    def manifest(self) -> PluginManifest: ...

    @abstractmethod
    def activate(self, controller: Any): ...

    def deactivate(self): ...


class AlgorithmPlugin(PluginBase):
    """Plugin that registers a classical ITK algorithm."""
    plugin_type = PluginType.ALGORITHM


class ModelPlugin(PluginBase):
    """Plugin that registers an AI model (external API endpoint).

    Unlike AlgorithmPlugin, this registers an API endpoint URL + adapter,
    not a local computation function.
    """
    plugin_type = PluginType.MODEL


class ToolPlugin(PluginBase):
    """Plugin that registers an interactive tool (paint, measure, annotate...)."""
    plugin_type = PluginType.TOOL


class ViewPlugin(PluginBase):
    """Plugin that adds a new view type (e.g., 4D time-series view)."""
    plugin_type = PluginType.VIEW


class ReaderPlugin(PluginBase):
    """Plugin that adds a new file format reader."""
    plugin_type = PluginType.READER
