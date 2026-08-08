"""Discovery and diagnostics for optional compiled MedAxis extensions."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Dict, Optional

# Directory that receives the compiled ``*.pyd`` modules (see native/*/CMakeLists.txt).
_NATIVE_DIR = Path(__file__).resolve().parent.parent / "medaxis" / "_native"

# DLL sub-directories probed under each MEDAXIS_CPP_PREFIX entry (conda-style layouts).
_DLL_SUBDIRS = ("Library/bin", "bin")

_configured = False


def _native_prefixes() -> list[str]:
    """C++ dependency prefixes: env var (``;``-separated) or user config.

    Priority: ``MEDAXIS_CPP_PREFIX`` env var, then the ``native.cpp_prefix``
    key of ``~/.medaxis/config.yaml``.  Each entry is probed for
    ``Library/bin`` / ``bin`` DLL directories.
    """
    value = os.environ.get("MEDAXIS_CPP_PREFIX", "")
    if not value:
        try:
            from utils.config import AppConfig

            value = AppConfig.instance().get("native", "cpp_prefix", "") or ""
        except Exception:
            value = ""
    return [p for p in value.split(";") if p]


def _configure_native_search() -> None:
    """Make compiled modules importable and their C++ runtime DLLs resolvable.

    - Adds ``medaxis/_native`` to ``sys.path`` so the ``*.pyd`` files produced by
      the CMake build can be imported as top-level modules.
    - Registers the runtime DLL directories of the C++ dependency prefix(es)
      (conda-forge environment and the ITK install prefix) with the Windows
      loader via ``os.add_dll_directory``.

    The prefix(es) come from ``MEDAXIS_CPP_PREFIX`` (env) or the
    ``native.cpp_prefix`` config key (see README "Native build").
    """
    global _configured
    if _configured:
        return
    _configured = True

    if _NATIVE_DIR.is_dir() and str(_NATIVE_DIR) not in sys.path:
        sys.path.insert(0, str(_NATIVE_DIR))

    if os.name != "nt":
        return

    # The pip VTK wheel keeps its runtime DLLs in ``vtk.libs`` — compiled
    # extensions (medaxis_bridge) link them by name, so the directory must be
    # visible to the Windows loader.
    try:
        import vtk

        _pip_vtk_libs = Path(vtk.__file__).resolve().parent / "vtk.libs"
        if _pip_vtk_libs.is_dir():
            os.add_dll_directory(str(_pip_vtk_libs))
    except Exception:
        pass

    for prefix in _native_prefixes():
        for sub in _DLL_SUBDIRS:
            dll_dir = os.path.join(prefix, sub)
            if os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except (OSError, ValueError):
                    pass


@dataclass(frozen=True)
class NativeExtensionStatus:
    """Result of loading one optional native extension."""

    name: str
    available: bool
    module: Optional[ModuleType] = None
    error: Optional[str] = None


class NativeExtensionRegistry:
    """Lazy, process-wide registry for compiled extension modules."""

    _instance: Optional["NativeExtensionRegistry"] = None

    def __init__(self) -> None:
        self._statuses: Dict[str, NativeExtensionStatus] = {}

    @classmethod
    def instance(cls) -> "NativeExtensionRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, name: str) -> NativeExtensionStatus:
        cached = self._statuses.get(name)
        if cached is not None:
            return cached

        _configure_native_search()
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError, ModuleNotFoundError) as exc:
            status = NativeExtensionStatus(name=name, available=False, error=str(exc))
        else:
            status = NativeExtensionStatus(name=name, available=True, module=module)

        self._statuses[name] = status
        return status

    def module(self, name: str) -> Optional[ModuleType]:
        return self.load(name).module

    def status(self) -> Dict[str, NativeExtensionStatus]:
        names = (
            "medaxis_bridge",
            "medaxis_itk",
            "medaxis_cgal",
            "medaxis_occ",
            "medaxis_radiomics",
        )
        return {name: self.load(name) for name in names}


def get_native_module(name: str) -> Optional[ModuleType]:
    """Return a compiled extension module, or ``None`` with lazy fallback."""

    return NativeExtensionRegistry.instance().module(name)
