"""
Action recorder — capture UI operations and generate medaxis API code.

Listens to the controller signals and records each operation as a
``medaxis.*`` call, Mimics-style macro recording.
"""
from __future__ import annotations

import textwrap
from typing import Callable, List


class ActionRecorder:
    """Records application actions into executable ``medaxis`` script lines."""

    def __init__(self) -> None:
        self._recording = False
        self._lines: List[str] = []
        self._listeners: List[Callable[[str], None]] = []

    # ------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        self._recording = True
        self._lines = ["# MedAxis action recording", "import medaxis"]
        self._emit("\n".join(self._lines))

    def stop(self) -> str:
        self._recording = False
        self._emit("")
        return "\n".join(self._lines)

    def clear(self) -> None:
        self._lines = []

    def script(self) -> str:
        return "\n".join(self._lines)

    def add_listener(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _emit(self, text: str) -> None:
        for fn in self._listeners:
            fn(text)

    # ------------------------------------------------------------------
    def _record(self, code: str) -> None:
        if not self._recording:
            return
        self._lines.append(code)
        self._emit("\n".join(self._lines))

    # -- action handlers (wired to controller signals) -----------------
    def on_volume_loaded(self, volume) -> None:
        self._record(f'vol = medaxis.io.open_nifti("{getattr(volume, "name", "volume")}") '
                     f"  # loaded {getattr(volume, 'dimensions', '')}")

    def on_label_created(self, label) -> None:
        self._record(f'label = medaxis.data.labels(id="{getattr(label, "id", "")}")')

    def on_algorithm_run(self, algorithm_id: str, params: dict) -> None:
        params_repr = textwrap.shorten(repr(params), width=80)
        self._record(f"medaxis.algorithms.run({algorithm_id!r}, **{params_repr})")

    def on_view_changed(self, axis: str, index: int) -> None:
        self._record(f"medaxis.views.set_slice(axis={axis!r}, index={index})")


# Global recorder (bound to the app in main_window).
recorder = ActionRecorder()


def bind_recorder_signals(controller) -> None:
    """Connect controller signals to the global action recorder."""
    controller.volume_loaded.connect(lambda v: recorder.on_volume_loaded(v))
    controller.label_created.connect(lambda label: recorder.on_label_created(label))
