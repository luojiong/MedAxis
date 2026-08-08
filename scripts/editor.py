"""
Script editor — minimal code editor with line numbers and run/stop.

Uses QScintilla when available (syntax highlighting); falls back to a
QPlainTextEdit otherwise.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QSyntaxHighlighter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:  # optional dependency
    _HAS_QSCINTILLA = False
    try:
        from QScintilla6.Qsci import QsciScintilla, QsciLexerPython  # type: ignore
        _HAS_QSCINTILLA = True
    except ImportError:
        pass
except Exception:  # pragma: no cover
    _HAS_QSCINTILLA = False

_KEYWORDS = {
    "def", "class", "return", "import", "from", "if", "elif", "else",
    "for", "while", "break", "continue", "pass", "try", "except",
    "finally", "with", "as", "lambda", "yield", "None", "True", "False",
    "and", "or", "not", "in", "is", "global", "nonlocal", "raise", "assert",
}


class _PythonHighlighter(QSyntaxHighlighter):
    """Lightweight Python syntax highlighting for the plain-text fallback."""

    def __init__(self, document) -> None:
        super().__init__(document)
        self._keyword_fmt = QTextCharFormat()
        self._keyword_fmt.setForeground(QColor("#569CD6"))
        self._keyword_fmt.setFontWeight(QFont.Weight.Bold)
        self._string_fmt = QTextCharFormat()
        self._string_fmt.setForeground(QColor("#CE9178"))
        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#6A9955"))
        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#B5CEA8"))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        import re

        for keyword in _KEYWORDS:
            for match in re.finditer(rf"\b{keyword}\b", text):
                self.setFormat(match.start(), match.end() - match.start(),
                               self._keyword_fmt)
        for match in re.finditer(r"#[^\n]*", text):
            self.setFormat(match.start(), match.end() - match.start(),
                           self._comment_fmt)
        for match in re.finditer(r"(\"\"\"[\s\S]*?\"\"\"|'[^']*'|\"[^\"]*\")", text):
            self.setFormat(match.start(), match.end() - match.start(),
                           self._string_fmt)
        for match in re.finditer(r"\b\d+(\.\d+)?\b", text):
            self.setFormat(match.start(), match.end() - match.start(),
                           self._number_fmt)


class ScriptEditor(QWidget):
    """Editor widget with syntax highlighting, run and save buttons."""

    def __init__(self, run_callback: Optional[Callable[[str], Any]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._run_callback = run_callback
        self._path: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        if _HAS_QSCINTILLA:
            self._editor = QsciScintilla(self)
            lexer = QsciLexerPython(self._editor)
            self._editor.setLexer(lexer)
            self._editor.setUtf8(True)
            self._editor.setAutoIndent(True)
        else:
            self._editor = QPlainTextEdit(self)
            self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self._highlighter = _PythonHighlighter(self._editor.document())

        self._editor.setPlaceholderText(
            "# MedAxis script — use the medaxis API, e.g.\n"
            "#   vols = medaxis.data.volumes()\n"
            "#   label = medaxis.algorithms.threshold(vols[0], lower=50, upper=300)")
        layout.addWidget(self._editor, 1)

        bar = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run (F5)", self)
        self._run_btn.clicked.connect(self._run)
        self._save_btn = QPushButton("Save", self)
        self._save_btn.clicked.connect(self.save)
        self._open_btn = QPushButton("Open…", self)
        self._open_btn.clicked.connect(self.open)
        bar.addWidget(self._run_btn)
        bar.addWidget(self._save_btn)
        bar.addWidget(self._open_btn)
        bar.addStretch(1)
        layout.addLayout(bar)

    # ------------------------------------------------------------------
    def set_code(self, code: str) -> None:
        self._editor.clear()
        self._editor.insertPlainText(code)

    def code(self) -> str:
        return self._editor.text() if _HAS_QSCINTILLA else self._editor.toPlainText()

    def run(self) -> Any:
        return self._run()

    def _run(self) -> Any:
        if self._run_callback is not None:
            return self._run_callback(self.code())
        return None

    def save(self, path: Optional[str] = None) -> Optional[str]:
        from PySide6.QtWidgets import QFileDialog

        path = path or self._path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save Script", "",
                                                  "Python (*.py)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.code())
            self._path = path
        return path

    def open(self, path: Optional[str] = None) -> Optional[str]:
        from PySide6.QtWidgets import QFileDialog

        path = path or self._path
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Script", "",
                                                  "Python (*.py)")
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                self.set_code(fh.read())
            self._path = path
        return path

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_F5:
            self._run()
            return
        super().keyPressEvent(event)
