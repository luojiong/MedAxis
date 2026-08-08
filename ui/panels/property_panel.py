"""Right-hand property panel for MedAxis.

:class:`PropertyPanel` is a :class:`QTabWidget`-based container with three
tabs:

* **Masks** — the :class:`LabelPanel` managing all label/mask objects;
* **Objects** — lists of generated meshes and measurements;
* **Export** — the :class:`ExportPanel` with quick-export buttons.

It is used as the right-hand side of the main window splitter.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .export_panel import ExportPanel
from .label_panel import LabelPanel


class PropertyPanel(QWidget):
    """Tabbed right-side panel: Masks | Objects | Export.

    Signals
    -------
    mesh_remove_requested(str)
        Emitted with the name of a mesh the user wants removed.
    measurement_remove_requested(int)
        Emitted with the row index of the measurement to remove.
    """

    mesh_remove_requested = Signal(str)
    measurement_remove_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PropertyPanel")

        self._tabs = QTabWidget(self)

        # -------------------------------------------------------------- #
        # Masks tab
        # -------------------------------------------------------------- #
        self.label_panel = LabelPanel(self)
        self._tabs.addTab(self.label_panel, "Masks")

        # -------------------------------------------------------------- #
        # Objects tab: mesh list + measurement list
        # -------------------------------------------------------------- #
        objects_widget = QWidget(self)
        objects_layout = QVBoxLayout(objects_widget)
        objects_layout.setContentsMargins(4, 4, 4, 4)

        objects_layout.addWidget(QLabel("Meshes:", objects_widget))
        self._mesh_list = QListWidget(objects_widget)
        objects_layout.addWidget(self._mesh_list)

        mesh_buttons = QHBoxLayout()
        self._mesh_remove_btn = QPushButton("Remove", objects_widget)
        self._mesh_remove_btn.clicked.connect(self._remove_selected_mesh)
        mesh_buttons.addWidget(self._mesh_remove_btn)
        mesh_buttons.addStretch(1)
        objects_layout.addLayout(mesh_buttons)

        objects_layout.addWidget(QLabel("Measurements:", objects_widget))
        self._measurement_tree = QTreeWidget(objects_widget)
        self._measurement_tree.setColumnCount(3)
        self._measurement_tree.setHeaderLabels(["Type", "Value", "Units"])
        self._measurement_tree.setRootIsDecorated(False)
        objects_layout.addWidget(self._measurement_tree, 1)

        meas_buttons = QHBoxLayout()
        self._meas_remove_btn = QPushButton("Remove", objects_widget)
        self._meas_remove_btn.clicked.connect(self._remove_selected_measurement)
        self._meas_clear_btn = QPushButton("Clear All", objects_widget)
        self._meas_clear_btn.clicked.connect(self.clear_measurements)
        meas_buttons.addWidget(self._meas_remove_btn)
        meas_buttons.addWidget(self._meas_clear_btn)
        meas_buttons.addStretch(1)
        objects_layout.addLayout(meas_buttons)

        self._tabs.addTab(objects_widget, "Objects")

        # -------------------------------------------------------------- #
        # Export tab
        # -------------------------------------------------------------- #
        self.export_panel = ExportPanel(self)
        self._tabs.addTab(self.export_panel, "Export")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

    # ------------------------------------------------------------------ #
    # Mesh list API
    # ------------------------------------------------------------------ #
    def add_mesh(self, name: str) -> None:
        self._mesh_list.addItem(QListWidgetItem(name))

    def remove_mesh(self, name: str) -> None:
        for row in range(self._mesh_list.count()):
            if self._mesh_list.item(row).text() == name:
                self._mesh_list.takeItem(row)
                return

    def clear_meshes(self) -> None:
        self._mesh_list.clear()

    def mesh_names(self):
        return [self._mesh_list.item(i).text()
                for i in range(self._mesh_list.count())]

    def _remove_selected_mesh(self) -> None:
        item = self._mesh_list.currentItem()
        if item is not None:
            name = item.text()
            self._mesh_list.takeItem(self._mesh_list.row(item))
            self.mesh_remove_requested.emit(name)

    # ------------------------------------------------------------------ #
    # Measurement list API
    # ------------------------------------------------------------------ #
    def add_measurement(self, mtype: str, value: str, units: str = "") -> None:
        item = QTreeWidgetItem([mtype, str(value), units])
        self._measurement_tree.addTopLevelItem(item)

    def clear_measurements(self) -> None:
        self._measurement_tree.clear()

    def measurement_count(self) -> int:
        return self._measurement_tree.topLevelItemCount()

    def _remove_selected_measurement(self) -> None:
        item = self._measurement_tree.currentItem()
        if item is None:
            return
        index = self._measurement_tree.indexOfTopLevelItem(item)
        self._measurement_tree.takeTopLevelItem(index)
        self.measurement_remove_requested.emit(index)

    # ------------------------------------------------------------------ #
    def set_tab(self, index_or_name) -> None:
        """Switch tabs by index or by name (``masks``/``objects``/``export``)."""
        names = {"masks": 0, "objects": 1, "export": 2}
        if isinstance(index_or_name, str):
            index_or_name = names.get(index_or_name.lower(), 0)
        self._tabs.setCurrentIndex(int(index_or_name))
