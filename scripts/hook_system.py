"""
Hook system — event hooks (on_volume_loaded, on_segmentation_done, ...).

Scripts can register callables that fire when the application emits the
corresponding signal, enabling automation (e.g. auto-segment on load).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

#: Hook names that scripts may subscribe to.
AVAILABLE_HOOKS = (
    "on_volume_loaded",
    "on_label_created",
    "on_segmentation_done",
    "on_view_changed",
    "on_project_changed",
    "on_session_closed",
)


class HookSystem:
    """Registry mapping hook names to lists of callables."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable]] = {name: [] for name in AVAILABLE_HOOKS}

    # ------------------------------------------------------------------
    def register(self, hook: str, fn: Callable) -> None:
        """Subscribe ``fn`` to ``hook``."""
        if hook not in self._hooks:
            raise KeyError(f"unknown hook '{hook}' (available: {AVAILABLE_HOOKS})")
        if fn not in self._hooks[hook]:
            self._hooks[hook].append(fn)

    def unregister(self, hook: str, fn: Callable) -> None:
        if hook in self._hooks and fn in self._hooks[hook]:
            self._hooks[hook].remove(fn)

    def fire(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Invoke every subscriber of ``hook``; failures are logged only."""
        for fn in list(self._hooks.get(hook, ())):
            try:
                fn(*args, **kwargs)
            except Exception:  # pragma: no cover - defensive
                logger.exception("hook %s failed", hook)

    def list_hooks(self) -> Dict[str, int]:
        return {name: len(fns) for name, fns in self._hooks.items()}

    def clear(self) -> None:
        for fns in self._hooks.values():
            fns.clear()


# Global instance bound to the application controller.
hook_system = HookSystem()


def bind_controller_signals(controller) -> None:
    """Connect the application signals to the global hook system."""
    controller.volume_loaded.connect(
        lambda volume: hook_system.fire("on_volume_loaded", volume))
    controller.label_created.connect(
        lambda label: hook_system.fire("on_label_created", label))
    if hasattr(controller, "project_changed"):
        controller.project_changed.connect(
            lambda project: hook_system.fire("on_project_changed", project))
