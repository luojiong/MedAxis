"""Export panel for MedAxis.

Offers one-click export buttons for the most common medical imaging formats
(NIfTI, NRRD, DICOM SEG, STL, OBJ, PLY, PNG, PDF), an output-directory
selector, and a scrolling export history.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

#: Supported export formats: key -> (button label, description).
EXPORT_FORMATS: Dict[str, str] = {
    "nifti": ("NIfTI (.nii/.nii.gz)", "Volume / mask as NIfTI"),
    "nrrd": ("NRRD (.nrrd)", "Volume / mask as NRRD"),
    "dicom_seg": ("DICOM SEG", "Segmentation as DICOM SEG object"),
    "stl": ("STL Mesh", "Surface mesh as STL"),
    "obj": ("OBJ Mesh", "Surface mesh as OBJ"),
    "ply": ("PLY Mesh", "Surface mesh as PLY"),
    "png": ("PNG Slice", "Current slice screenshot as PNG"),
    "pdf": ("PDF Report", "Report / measurement summary as PDF"),
}


class ExportPanel(QWidget):
    """Quick-export controls plus export history.

    Signals
    -------
    export_requested(str)
        Emitted with the format key (see :data:`EXPORT_FORMATS`) when the
        user clicks an export button.
    output_dir_changed(str)
        Emitted when the chosen output directory changes.
    """

    export_requested = Signal(str)
    output_dir_changed = Signal(str)

    MAX_HISTORY = 200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExportPanel")
        self._output_dir: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -------------------------------------------------------------- #
        # Quick export buttons
        # -------------------------------------------------------------- #
        formats_group = QGroupBox("Quick Export", self)
        grid = QGridLayout(formats_group)
        grid.setSpacing(4)

        self._buttons: Dict[str, QPushButton] = {}
        keys = list(EXPORT_FORMATS.keys())
        for i, key in enumerate(keys):
            label, description = EXPORT_FORMATS[key]
            btn = QPushButton(label, formats_group)
            btn.setToolTip(description)
            btn.clicked.connect(lambda _=False, k=key: self.export_requested.emit(k))
            grid.addWidget(btn, i // 2, i % 2)
            self._buttons[key] = btn

        layout.addWidget(formats_group)

        # -------------------------------------------------------------- #
        # Output directory selector
        # -------------------------------------------------------------- #
        dir_group = QGroupBox("Output Directory", self)
        dir_layout = QHBoxLayout(dir_group)
        self._dir_label = QLabel("<i>not set</i>", dir_group)
        self._dir_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...", dir_group)
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self._dir_label, 1)
        dir_layout.addWidget(browse_btn)
        layout.addWidget(dir_group)

        # -------------------------------------------------------------- #
        # Export history
        # -------------------------------------------------------------- #
        history_group = QGroupBox("Export History", self)
        history_layout = QVBoxLayout(history_group)
        self._history_list = QListWidget(history_group)
        self._history_list.setSelectionMode(QListWidget.NoSelection)
        clear_btn = QPushButton("Clear History", history_group)
        clear_btn.clicked.connect(self._history_list.clear)
        history_layout.addWidget(self._history_list, 1)
        history_layout.addWidget(clear_btn)
        layout.addWidget(history_group, 1)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def output_dir(self) -> str:
        return self._output_dir

    def set_output_dir(self, path: str) -> None:
        if path == self._output_dir:
            return
        self._output_dir = path
        self._dir_label.setText(path if path else "<i>not set</i>")
        self.output_dir_changed.emit(path)

    def add_history_entry(self, text: str) -> None:
        """Insert *text* at the top of the export history."""
        item = QListWidgetItem(text)
        self._history_list.insertItem(0, item)
        while self._history_list.count() > self.MAX_HISTORY:
            self._history_list.takeItem(self._history_list.count() - 1)

    def set_export_enabled(self, enabled: bool) -> None:
        for btn in self._buttons.values():
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", self._output_dir)
        if path:
            self.set_output_dir(path)
