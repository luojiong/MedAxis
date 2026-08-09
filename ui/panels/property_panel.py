"""Compact right-side object and display management area."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget,
)

from .export_panel import ExportPanel
from .label_panel import LabelPanel
from ..glass import GlassPanel


class _ObjectSection(GlassPanel):
    """Reusable tabbed object section with a quiet empty state."""

    def __init__(self, tabs: list[str], columns: list[str], parent=None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget(self)
        for title in tabs:
            table = QTableWidget(0, len(columns), self)
            table.setHorizontalHeaderLabels(columns)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setAlternatingRowColors(True)
            table.setShowGrid(False)
            table.setSortingEnabled(True)
            table.setMinimumHeight(82)
            table.horizontalHeader().setStretchLastSection(True)
            table.setRowCount(1)
            empty = QLabel("No objects", table)
            empty.setObjectName("EmptyState")
            empty.setAlignment(Qt.AlignCenter)
            table.setCellWidget(0, 0, empty)
            self.tabs.addTab(table, title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        actions = QHBoxLayout()
        for icon, tip in (("+", "Add object"), ("Edit", "Edit object"),
                          ("Copy", "Duplicate object"), ("Delete", "Delete object"),
                          ("Export", "Export object")):
            btn = QToolButton(self)
            btn.setText(icon)
            btn.setToolTip(tip)
            actions.addWidget(btn)
        actions.addStretch(1)
        layout.addLayout(actions)


class _DisplaySection(GlassPanel):
    """Display parameter tabs matching the lower-right reference panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget(self)
        contrast = QWidget(self)
        contrast_layout = QVBoxLayout(contrast)
        contrast_layout.setContentsMargins(8, 8, 8, 6)
        histogram = QLabel("▁▂▃▅▇▆▅▄▃▂▁", contrast)
        histogram.setObjectName("HistogramPreview")
        histogram.setAlignment(Qt.AlignCenter)
        histogram.setMinimumHeight(74)
        contrast_layout.addWidget(histogram)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Grayscale"))
        controls.addWidget(QLabel("Custom"))
        controls.addWidget(QLabel("Min  -1023"))
        controls.addWidget(QLabel("Max  3071"))
        contrast_layout.addLayout(controls)
        self.tabs.addTab(contrast, "Contrast")
        for title, message in (("Volume Rendering", "Preset: Bone\nOpacity and sampling"),
                               ("Clipping", "Clipping plane disabled"),
                               ("Snapping", "Snapping target: None")):
            page = QLabel(message, self)
            page.setObjectName("EmptyState")
            page.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(page, title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)


class PropertyPanel(QWidget):
    """Right rail: masks, model objects, meshes and display parameters."""

    mesh_remove_requested = Signal(str)
    measurement_remove_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PropertyPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._mask_tabs = QTabWidget(self)
        self._mask_tabs.setObjectName("GlassBlockMasks")
        self.label_panel = LabelPanel(self)
        self._mask_tabs.addTab(self.label_panel, "Masks")
        self._mask_tabs.addTab(self._empty_page("No measurements"), "Measurements")
        self._mask_tabs.addTab(self._empty_page("No annotations"), "Annotations")
        layout.addWidget(self._mask_tabs, 5)

        # Retain the public export-panel hook used by MainWindow/controller
        # wiring. The export UI remains available through the command layer.
        self.export_panel = ExportPanel(self)
        self.export_panel.setVisible(False)

        self._object_section = _ObjectSection(
            ["Parts", "Analysis Objects", "Reslice Objects", "Soft tissue"],
            ["Name", "Visible", "Transform", "Quality"], self)
        self._object_section.setObjectName("GlassBlockObjects")
        layout.addWidget(self._object_section, 3)

        self._mesh_section = _ObjectSection(
            ["STLs", "Polylines", "Simulation Objects", "FEA Mesh"],
            ["Name", "Visible", "Triangles", "Transform"], self)
        self._mesh_section.setObjectName("GlassBlockMeshes")
        layout.addWidget(self._mesh_section, 3)

        self._display_section = _DisplaySection(self)
        self._display_section.setObjectName("GlassBlockDisplay")
        layout.addWidget(self._display_section, 3)

        self._mesh_names: list[str] = []

    @staticmethod
    def _empty_page(text: str) -> QLabel:
        page = QLabel(text)
        page.setObjectName("EmptyState")
        page.setAlignment(Qt.AlignCenter)
        return page

    def add_mesh(self, name: str) -> None:
        self._mesh_names.append(name)

    def remove_mesh(self, name: str) -> None:
        if name in self._mesh_names:
            self._mesh_names.remove(name)

    def clear_meshes(self) -> None:
        self._mesh_names.clear()

    def mesh_names(self):
        return list(self._mesh_names)

    def add_measurement(self, mtype: str, value: str, units: str = "") -> None:
        self._mask_tabs.setCurrentIndex(1)

    def clear_measurements(self) -> None:
        pass

    def measurement_count(self) -> int:
        return 0

    def set_tab(self, index_or_name) -> None:
        names = {"masks": 0, "measurements": 1, "annotations": 2,
                 "objects": 0, "export": 0}
        if isinstance(index_or_name, str):
            index_or_name = names.get(index_or_name.lower(), 0)
        self._mask_tabs.setCurrentIndex(int(index_or_name))
