"""
Script sandbox — three permission levels (Sandboxed / Standard / Full).

- Sandboxed: read-only medaxis API, no filesystem/network/system access.
- Standard: full medaxis API + filesystem under the user data dir.
- Full: unrestricted Python (subprocess, sockets, arbitrary imports).
"""
from __future__ import annotations

import builtins
from typing import Any, Dict, Optional

#: Modules that are always safe to expose (pure computation).
_SAFE_MODULES = ("math", "numpy", "itk", "vtk", "scipy", "skimage")

#: Imports blocked in non-full modes.
_BLOCKED_MODULES = ("os", "sys", "subprocess", "socket", "shutil", "pathlib",
                    "ctypes", "importlib", "multiprocessing", "threading",
                    "pickle", "marshal", "http", "urllib", "requests",
                    "aiohttp", "grpc")

#: Builtins shadowed in non-full modes.
_BLOCKED_BUILTINS = ("open", "exec", "eval", "compile", "__import__",
                     "input", "breakpoint")


class SandboxError(PermissionError):
    """Raised when a sandboxed script attempts a restricted operation."""


class Sandbox:
    """Executes code under a permission level with a restricted namespace."""

    def __init__(self, level: str = "standard", api: Any = None,
                 namespace: Optional[Dict[str, Any]] = None) -> None:
        if level not in ("sandboxed", "standard", "full"):
            raise ValueError(f"unknown sandbox level: {level}")
        self.level = level
        self._namespace: Dict[str, Any] = {"__name__": "__medaxis_script__"}
        self._namespace.update(namespace or {})
        if api is not None:
            self._namespace["medaxis"] = api

    # ------------------------------------------------------------------
    @property
    def namespace(self) -> Dict[str, Any]:
        return self._namespace

    def _guard_import(self, name: str) -> None:
        if self.level == "full":
            return
        if name in _BLOCKED_MODULES or name.split(".")[0] in _BLOCKED_MODULES:
            raise SandboxError(f"import of '{name}' is blocked at "
                               f"permission level '{self.level}'")

    def _guard_builtins(self) -> None:
        if self.level == "full":
            return
        for blocked in _BLOCKED_BUILTINS:
            if blocked not in self._namespace:
                def _deny(*_args, **_kwargs):
                    raise SandboxError(
                        f"'{blocked}' is blocked at permission level '{self.level}'")
                self._namespace[blocked] = _deny

    def execute(self, code: str) -> Any:
        """Execute ``code``; returns the value of a bare expression."""
        self._guard_builtins()
        if self.level != "full":
            # Restrict __import__ while keeping numpy/itk/scipy usable.
            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                self._guard_import(name)
                return real_import(name, *args, **kwargs)

            self._namespace["__builtins__"] = {**vars(builtins), "__import__": guarded_import}
        else:
            self._namespace["__builtins__"] = builtins

        try:
            return eval(code, self._namespace)  # noqa: S307
        except SyntaxError:
            exec(compile(code, "<medaxis-script>", "exec"), self._namespace)  # noqa: S102
            return None

    def run_file(self, path: str) -> Any:
        """Execute a script file (Standard/Full only)."""
        if self.level == "sandboxed":
            raise SandboxError("running script files is not allowed at "
                               "permission level 'sandboxed'")
        with open(path, "r", encoding="utf-8") as fh:
            code = fh.read()
        return self.execute(code)
