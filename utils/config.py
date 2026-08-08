"""MedAxis — Application Configuration.

Singleton AppConfig loading/saving a YAML file at ~/.medaxis/config.yaml.
Falls back to an internal defaults dict when PyYAML is unavailable.

Sections: paths, rendering, processing, ai, ui, plugins.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".medaxis")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "paths": {
        "last_open_dir": "",
        "last_save_dir": "",
        "temp_dir": os.path.join(CONFIG_DIR, "tmp"),
        "cache_dir": os.path.join(CONFIG_DIR, "cache"),
        "plugin_dirs": [],
    },
    "rendering": {
        "backend": "vtk",
        "gpu_accelerated": True,
        "max_texture_memory_mb": 4096,
        "default_interpolation": "linear",
        "volume_render_quality": "medium",   # low | medium | high
        "multiplanar_layout": "2x2",
        "background_color": [0, 0, 0],
        "default_colormap": "grayscale",
        "slice_thickness_display": True,
    },
    "processing": {
        "num_threads": 0,          # 0 = auto-detect
        "resample_spacing": [1.0, 1.0, 1.0],
        "smoothing_sigma": 1.0,
        "auto_orient": True,
        "keep_original_dtype": False,
    },
    "ai": {
        "enabled": True,
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "base_url": "",
        "timeout_seconds": 60,
        "max_retries": 3,
        "stream": True,
        "temperature": 0.2,
    },
    "ui": {
        "theme": "dark",
        "language": "en",
        "window_geometry": None,
        "window_state": None,
        "recent_projects_limit": 10,
        "confirm_on_close": True,
        "auto_save_interval_seconds": 0,   # 0 = disabled
        "status_bar_visible": True,
    },
    "plugins": {
        "enabled": True,
        "auto_load": True,
        "disabled_plugins": [],
        "trusted_sources": [],
    },
}


class AppConfig:
    """Singleton application configuration backed by YAML."""

    _instance: Optional["AppConfig"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = copy.deepcopy(DEFAULT_CONFIG)
        self.path: str = CONFIG_PATH
        self._dirty = False

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "AppConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cfg = cls()
                    cfg.load()
                    cls._instance = cfg
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self, path: Optional[str] = None) -> None:
        """Load config from disk, merging over defaults."""
        if path:
            self.path = path
        if not os.path.exists(self.path):
            logger.info("No config file at %s; using defaults.", self.path)
            return

        data = self._read_yaml(self.path)
        if not isinstance(data, dict):
            logger.warning("Config file is not a mapping; using defaults.")
            return

        for section, values in data.items():
            if section in self._data and isinstance(values, dict):
                self._data[section].update(values)
            else:
                self._data[section] = values
        logger.info("Loaded config from %s", self.path)

    def save(self, path: Optional[str] = None) -> bool:
        """Persist current config to disk."""
        target = path or self.path
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            self._write_yaml(target, self._data)
            self._dirty = False
            logger.debug("Saved config to %s", target)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save config to %s", target)
            return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a config value. Supports dotted keys, e.g. 'ai.model'."""
        if "." in section and key is None:
            section, key = section.split(".", 1)
        return self._data.get(section, {}).get(key, default)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Get by 'section.key' dotted path."""
        if "." not in dotted:
            return default
        section, key = dotted.split(".", 1)
        return self.get(section, key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self._data.setdefault(section, {})[key] = value
        self._dirty = True

    def section(self, name: str) -> Dict[str, Any]:
        """Return a whole section dict (live reference)."""
        return self._data.setdefault(name, {})

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # YAML I/O with graceful fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _read_yaml(path: str) -> Any:
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed; reading config as JSON fallback.")
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    @staticmethod
    def _write_yaml(path: str, data: Dict[str, Any]) -> None:
        try:
            import yaml  # type: ignore
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False,
                               sort_keys=False, allow_unicode=True)
        except ImportError:
            logger.warning("PyYAML not installed; writing config as JSON fallback.")
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
