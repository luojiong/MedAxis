"""
Script console — interactive REPL backed by the medaxis API + sandbox.
"""
from __future__ import annotations

import traceback
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .sandbox import Sandbox, SandboxError


class ScriptConsole(QWidget):
    """Interactive Python REPL with prompt, history and sandboxed execution.

    The ``medaxis`` facade and the ``hook`` system are injected into the
    namespace; the execution level is configurable (see :class:`Sandbox`).
    """

    def __init__(self, api: Any = None, level: str = "standard",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sandbox = Sandbox(level=level, api=api)
        self._history: list[str] = []
        self._history_pos = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._output = QPlainTextEdit(self)
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(10000)
        self._output.setPlaceholderText(
            "MedAxis script console — type Python, e.g.  medaxis.data.volumes()")
        layout.addWidget(self._output, 1)

        row = QHBoxLayout()
        self._prompt = QLineEdit(">>>", self)
        self._prompt.setFixedWidth(28)
        self._prompt.setReadOnly(True)
        self._input = QLineEdit(self)
        self._input.setPlaceholderText("Enter Python code, press Enter")
        self._input.returnPressed.connect(self._execute_line)
        row.addWidget(self._prompt)
        row.addWidget(self._input, 1)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    def append_context(self, context: dict) -> None:
        self._sandbox.namespace.update(context)

    def print(self, text: str) -> None:
        self._output.appendPlainText(str(text))

    def execute(self, code: str) -> Any:
        """Run ``code`` in the sandboxed namespace and print the result."""
        self._output.appendPlainText(f">>> {code}")
        try:
            result = self._sandbox.execute(code)
        except SandboxError as exc:
            self._output.appendPlainText(f"SandboxError: {exc}")
            return None
        except Exception:
            self._output.appendPlainText(traceback.format_exc(limit=3))
            return None
        if result is not None:
            self._output.appendPlainText(repr(result))
        return result

    def _execute_line(self) -> None:
        code = self._input.text().strip()
        if not code:
            return
        self._input.clear()
        self._history.append(code)
        self._history_pos = len(self._history)
        self.execute(code)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            if self._history:
                if event.key() == Qt.Key_Up:
                    self._history_pos = max(0, self._history_pos - 1)
                else:
                    self._history_pos = min(len(self._history), self._history_pos + 1)
                if 0 <= self._history_pos < len(self._history):
                    self._input.setText(self._history[self._history_pos])
                else:
                    self._input.clear()
                self._input.setCursorPosition(len(self._input.text()))
                return
        super().keyPressEvent(event)
