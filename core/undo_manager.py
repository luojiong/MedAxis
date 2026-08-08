"""UndoManager — singleton undo/redo stack with sparse label diffs.

Wraps Qt's ``QUndoStack`` when PyQt6 is available; falls back to a pure
Python stack otherwise so the module stays importable headless.

Memory model:
    * :class:`EditLabelCommand` stores only the changed voxels inside a
      bounding box (sparse diff), not whole-array copies.
    * The stack enforces ``MAX_MEMORY_BYTES`` (500 MB) and ``MAX_STEPS``
      (200) by dropping the oldest commands.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "UndoManager",
    "EditLabelCommand",
    "DeleteLabelCommand",
    "AddMeasurementCommand",
    "ChangeViewCommand",
]

MAX_MEMORY_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_STEPS = 200

# ---------------------------------------------------------------------------
# Qt availability detection
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on environment
    from PySide6.QtCore import QObject, Signal as pyqtSignal
    from PySide6.QtGui import QUndoCommand, QUndoStack

    _HAS_QT = True
except ImportError:
    try:
        from PyQt6.QtCore import QObject, pyqtSignal
        from PyQt6.QtGui import QUndoCommand, QUndoStack
        _HAS_QT = True
    except ImportError:
        _HAS_QT = False

        class QObject:  # type: ignore[no-redef]
            """Minimal QObject stand-in."""

        def pyqtSignal(*_args, **_kwargs):  # type: ignore[no-redef]
            return None

        class QUndoCommand:  # type: ignore[no-redef]
            """Minimal QUndoCommand stand-in."""

            def __init__(self, text: str = "", parent=None):
                self._text = text

            def redo(self) -> None:  # noqa: D102
                raise NotImplementedError

            def undo(self) -> None:  # noqa: D102
                raise NotImplementedError

            def setText(self, text: str) -> None:  # noqa: N802
                self._text = text

            def text(self) -> str:  # noqa: D102
                return self._text


# ---------------------------------------------------------------------------
# Simple signal implementation (used for the Qt-less fallback path too)
# ---------------------------------------------------------------------------
class _Signal:
    """Tiny multi-listener signal."""

    def __init__(self) -> None:
        self._slots: list[Callable[..., None]] = []

    def connect(self, slot: Callable[..., None]) -> None:
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot: Callable[..., None]) -> None:
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args: Any) -> None:
        for slot in list(self._slots):
            slot(*args)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
class EditLabelCommand(QUndoCommand):
    """Paint/erase voxels in a label map, storing a sparse diff.

    Only the bounding box of the edited region plus old/new voxel values
    are stored, keeping memory usage proportional to the stroke size.
    """

    def __init__(
        self,
        label_array: np.ndarray,
        bbox_min: tuple[int, int, int],
        bbox_max: tuple[int, int, int],
        old_values: np.ndarray,
        new_values: np.ndarray,
        description: str = "Edit label",
    ) -> None:
        super().__init__(description)
        self._array = label_array
        self._min = tuple(int(v) for v in bbox_min)
        self._max = tuple(int(v) for v in bbox_max)  # exclusive
        self._old = np.ascontiguousarray(old_values)
        self._new = np.ascontiguousarray(new_values)
        self._bytes = self._old.nbytes + self._new.nbytes

    @property
    def memory_bytes(self) -> int:
        """Approximate memory footprint of the stored diff."""
        return self._bytes

    def _slice(self) -> tuple[slice, slice, slice]:
        return (
            slice(self._min[0], self._max[0]),
            slice(self._min[1], self._max[1]),
            slice(self._min[2], self._max[2]),
        )

    def redo(self) -> None:  # noqa: D102
        self._array[self._slice()] = self._new

    def undo(self) -> None:  # noqa: D102
        self._array[self._slice()] = self._old

    def mergeWith(self, other: "QUndoCommand") -> bool:  # noqa: N802
        """Never auto-merge; strokes are kept separate for clarity."""
        return False


class DeleteLabelCommand(QUndoCommand):
    """Remove a label entry (and optionally its array data) from a registry.

    ``registry`` is any object with ``__getitem__`` / ``__setitem__`` /
    ``__delitem__`` semantics keyed by label id (e.g. a dict of loaded
    labels). The removed item is kept alive for undo.
    """

    def __init__(self, registry: Any, label_id: str, description: str = "Delete label") -> None:
        super().__init__(description)
        self._registry = registry
        self._label_id = label_id
        self._removed: Any = None
        arr = getattr(registry.get(label_id), "array", None) if hasattr(registry, "get") else None
        self._bytes = int(arr.nbytes) if arr is not None else 0

    @property
    def memory_bytes(self) -> int:
        """Approximate memory footprint of the held label data."""
        return self._bytes

    def redo(self) -> None:  # noqa: D102
        self._removed = self._registry[self._label_id]
        del self._registry[self._label_id]

    def undo(self) -> None:  # noqa: D102
        self._registry[self._label_id] = self._removed


class AddMeasurementCommand(QUndoCommand):
    """Add (redo) or remove (undo) a measurement in a list."""

    def __init__(self, measurements: list, measurement: Any, description: str = "Add measurement") -> None:
        super().__init__(description)
        self._measurements = measurements
        self._measurement = measurement

    @property
    def memory_bytes(self) -> int:
        """Measurements are tiny; report a nominal constant."""
        return 256

    def redo(self) -> None:  # noqa: D102
        if self._measurement not in self._measurements:
            self._measurements.append(self._measurement)

    def undo(self) -> None:  # noqa: D102
        if self._measurement in self._measurements:
            self._measurements.remove(self._measurement)


class ChangeViewCommand(QUndoCommand):
    """Store a change to a view/camera configuration dict."""

    def __init__(
        self,
        view_state: dict[str, Any],
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        description: str = "Change view",
    ) -> None:
        super().__init__(description)
        self._view_state = view_state
        self._old = dict(old_state)
        self._new = dict(new_state)

    @property
    def memory_bytes(self) -> int:
        """Nominal constant; view states are small dicts."""
        return 512

    def redo(self) -> None:  # noqa: D102
        self._view_state.clear()
        self._view_state.update(self._new)

    def undo(self) -> None:  # noqa: D102
        self._view_state.clear()
        self._view_state.update(self._old)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class UndoManager(QObject):
    """Application-wide undo/redo manager (singleton).

    Use :meth:`instance` to obtain the shared object.
    """

    if _HAS_QT:
        undo_stack_changed = pyqtSignal()
    else:
        undo_stack_changed = _Signal()

    _instance: Optional["UndoManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()
        if _HAS_QT:
            self._stack = QUndoStack()
            self._stack.indexChanged.connect(lambda *_: self._emit_changed())
        else:
            self._stack = None
            self._undo: list[QUndoCommand] = []
            self._redo: list[QUndoCommand] = []
        self._current_bytes = 0

    # -- singleton ------------------------------------------------------

    @classmethod
    def instance(cls) -> "UndoManager":
        """Return the shared :class:`UndoManager`, creating it on first use."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # -- internal ---------------------------------------------------------

    def _emit_changed(self) -> None:
        if _HAS_QT:
            self.undo_stack_changed.emit()
        else:
            self.undo_stack_changed.emit()  # type: ignore[union-attr]

    def _enforce_limits(self, incoming_bytes: int) -> None:
        """Drop oldest undo commands until under the memory/step limits."""
        if _HAS_QT:
            # QUndoStack does not expose popping old commands; set a
            # conservative upper bound instead.
            self._stack.setUndoLimit(MAX_STEPS)
            return

        while self._undo and (
            len(self._undo) + 1 > MAX_STEPS
            or self._current_bytes + incoming_bytes > MAX_MEMORY_BYTES
        ):
            dropped = self._undo.pop(0)
            self._current_bytes -= getattr(dropped, "memory_bytes", 0)

    # -- public API ------------------------------------------------------

    def push(self, command: QUndoCommand) -> None:
        """Execute and record a command."""
        mem = getattr(command, "memory_bytes", 0)
        if _HAS_QT:
            self._stack.push(command)  # calls redo()
        else:
            self._enforce_limits(mem)
            command.redo()
            self._undo.append(command)
            self._redo.clear()
            self._current_bytes += mem
        self._emit_changed()

    def undo(self) -> None:
        """Undo the most recent command, if any."""
        if _HAS_QT:
            self._stack.undo()
        elif self._undo:
            cmd = self._undo.pop()
            cmd.undo()
            self._redo.append(cmd)
            self._current_bytes -= getattr(cmd, "memory_bytes", 0)
        self._emit_changed()

    def redo(self) -> None:
        """Redo the most recently undone command, if any."""
        if _HAS_QT:
            self._stack.redo()
        elif self._redo:
            cmd = self._redo.pop()
            cmd.redo()
            self._undo.append(cmd)
            self._current_bytes += getattr(cmd, "memory_bytes", 0)
        self._emit_changed()

    def clear(self) -> None:
        """Remove all commands from the stack."""
        if _HAS_QT:
            self._stack.clear()
        else:
            self._undo.clear()
            self._redo.clear()
            self._current_bytes = 0
        self._emit_changed()

    @property
    def can_undo(self) -> bool:
        """Whether an undo operation is available."""
        return self._stack.canUndo() if _HAS_QT else bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether a redo operation is available."""
        return self._stack.canRedo() if _HAS_QT else bool(self._redo)

    @property
    def count(self) -> int:
        """Number of commands on the undo stack."""
        return self._stack.count() if _HAS_QT else len(self._undo)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory used by stored command data (fallback stack)."""
        if _HAS_QT:
            return -1  # not trackable through QUndoStack
        return self._current_bytes
