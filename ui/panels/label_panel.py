"""Label (mask) management panel for MedAxis.

Displays every :class:`LabelData` item in a tree view with the columns::

    Color | Name | Visible | Volume (ml) | Source

A bottom toolbar offers New / Delete / Duplicate / Export and a rich
right-click context menu covers rename, colour change, hole filling, boolean
operations, 3D surface rebuild and statistics.  Rows can be reordered by
dragging.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QColorDialog, QHBoxLayout, QHeaderView, QMenu,
    QToolButton, QTreeView, QVBoxLayout, QWidget,
)

# Column indices.
COL_COLOR = 0
COL_NAME = 1
COL_VISIBLE = 2
COL_VOLUME = 3
COL_SOURCE = 4

_DEFAULT_COLORS = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#008080",
)


class LabelPanel(QWidget):
    """Panel listing and managing all segmentation labels/masks.

    Signals
    -------
    label_selected(int)
        Row index of the newly selected label.
    label_visibility_changed(int, bool)
        Row index and new visibility state.
    label_color_changed(int, QColor)
        Row index and the new colour.
    label_added(int)
        A new label row was created (via the *New* button).
    label_removed(int)
        A label row was removed.
    label_reordered(list[int])
        New row order expressed as a list of original indices.
    label_rename_requested(int)
        Context-menu *Rename* invoked for a row.
    label_fill_holes_requested(int)
        Context-menu *Fill Holes* invoked for a row.
    label_boolean_requested(int, str)
        Boolean operation requested: payload is one of ``"union"``,
        ``"intersect"``, ``"subtract"``.
    label_rebuild_requested(int)
        *Rebuild 3D* invoked for a row.
    label_statistics_requested(int)
        *Statistics* invoked for a row.
    label_export_requested(int)
        *Export* invoked for a row.
    label_duplicate_requested(int)
        *Duplicate* invoked for a row.
    """

    label_selected = Signal(int)
    label_visibility_changed = Signal(int, bool)
    label_color_changed = Signal(int, QColor)
    label_added = Signal(int)
    label_removed = Signal(int)
    label_reordered = Signal(list)
    label_rename_requested = Signal(int)
    label_fill_holes_requested = Signal(int)
    label_boolean_requested = Signal(int, str)
    label_rebuild_requested = Signal(int)
    label_statistics_requested = Signal(int)
    label_export_requested = Signal(int)
    label_duplicate_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LabelPanel")

        self._model = QStandardItemModel(self)
        self._model.setHorizontalHeaderLabels(
            ["Color", "Name", "Visible", "Volume (ml)", "Source"])

        self._view = QTreeView(self)
        self._view.setModel(self._model)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._view.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed)
        self._view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_context_menu)

        # Drag & drop reordering.
        self._view.setDragDropMode(QAbstractItemView.InternalMove)
        self._view.setDefaultDropAction(Qt.MoveAction)
        self._view.setDragDropOverwriteMode(False)

        header = self._view.header()
        header.setSectionResizeMode(COL_COLOR, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_VOLUME, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeToContents)

        self._view.clicked.connect(self._on_clicked)
        self._view.pressed.connect(self._on_pressed)
        self._model.itemChanged.connect(self._on_item_changed)
        sel = self._view.selectionModel()
        sel.selectionChanged.connect(self._on_selection_changed)

        # Bottom toolbar.
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        self._btn_new = QToolButton(self)
        self._btn_new.setText("New")
        self._btn_new.clicked.connect(self.add_label)
        self._btn_delete = QToolButton(self)
        self._btn_delete.setText("Delete")
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_duplicate = QToolButton(self)
        self._btn_duplicate.setText("Duplicate")
        self._btn_duplicate.clicked.connect(self._duplicate_selected)
        self._btn_export = QToolButton(self)
        self._btn_export.setText("Export")
        self._btn_export.clicked.connect(self._export_selected)
        for btn in (self._btn_new, self._btn_delete,
                    self._btn_duplicate, self._btn_export):
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            toolbar.addWidget(btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view, 1)
        layout.addLayout(toolbar)

        # Track drag start row so we can report reorders.
        self._drag_start_row: int = -1
        self._next_color_index = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_label(
        self,
        name: str = "New Label",
        color: Optional[QColor] = None,
        volume_ml: float = 0.0,
        source: str = "Manual",
        visible: bool = True,
        emit: bool = True,
    ) -> int:
        """Append a label row; returns the new row index."""
        if color is None:
            color = QColor(_DEFAULT_COLORS[self._next_color_index
                                           % len(_DEFAULT_COLORS)])
            self._next_color_index += 1
        self._model.appendRow(self._make_row(name, color, volume_ml,
                                             source, visible))
        row = self._model.rowCount() - 1
        self._view.setCurrentIndex(self._model.index(row, COL_NAME))
        if emit:
            self.label_added.emit(row)
        return row

    def label_count(self) -> int:
        return self._model.rowCount()

    def selected_row(self) -> int:
        indexes = self._view.selectionModel().selectedRows()
        return indexes[0].row() if indexes else -1

    def select_row(self, row: int) -> None:
        if 0 <= row < self._model.rowCount():
            self._view.setCurrentIndex(self._model.index(row, COL_NAME))

    def label_name(self, row: int) -> str:
        item = self._model.item(row, COL_NAME)
        return item.text() if item else ""

    def label_color(self, row: int) -> QColor:
        item = self._model.item(row, COL_COLOR)
        return item.data(Qt.UserRole) or QColor("white")

    def label_visible(self, row: int) -> bool:
        item = self._model.item(row, COL_VISIBLE)
        return item.checkState() == Qt.Checked if item else True

    def remove_label(self, row: int) -> None:
        if 0 <= row < self._model.rowCount():
            self._model.removeRow(row)
            self.label_removed.emit(row)

    def set_volume(self, row: int, volume_ml: float) -> None:
        item = self._model.item(row, COL_VOLUME)
        if item is not None:
            item.setText(f"{volume_ml:.2f}")

    def clear(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    # ------------------------------------------------------------------ #
    # Row construction
    # ------------------------------------------------------------------ #
    def _make_row(self, name: str, color: QColor, volume_ml: float,
                  source: str, visible: bool) -> List[QStandardItem]:
        color_item = QStandardItem()
        color_item.setBackground(QBrush(color))
        color_item.setData(color, Qt.UserRole)
        color_item.setEditable(False)
        color_item.setToolTip("Click to change colour")

        name_item = QStandardItem(name)
        name_item.setEditable(True)
        name_item.setData(color, Qt.UserRole)  # paired colour for lookups

        vis_item = QStandardItem()
        vis_item.setCheckable(True)
        vis_item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        vis_item.setEditable(False)

        vol_item = QStandardItem(f"{volume_ml:.2f}")
        vol_item.setEditable(False)

        src_item = QStandardItem(source)
        src_item.setEditable(False)

        for item in (color_item, name_item, vis_item, vol_item, src_item):
            item.setFlags(item.flags() | Qt.ItemIsDragEnabled
                          | Qt.ItemIsDropEnabled)
        return [color_item, name_item, vis_item, vol_item, src_item]

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_clicked(self, index: QModelIndex) -> None:
        row = index.row()
        if index.column() == COL_COLOR:
            current = self.label_color(row)
            color = QColorDialog.getColor(current, self, "Label colour")
            if color.isValid():
                self._model.item(row, COL_COLOR).setBackground(QBrush(color))
                self._model.item(row, COL_COLOR).setData(color, Qt.UserRole)
                self.label_color_changed.emit(row, color)

    def _on_pressed(self, index: QModelIndex) -> None:
        # Remember row at drag start for reorder reporting.
        self._drag_start_row = index.row()

    def _on_item_changed(self, item: QStandardItem) -> None:
        row = item.row()
        if item.column() == COL_VISIBLE:
            visible = item.checkState() == Qt.Checked
            self.label_visibility_changed.emit(row, visible)
        elif item.column() == COL_NAME:
            # Name edited inline.
            pass
        # Detect drop-completed reorder: rows moved internally.
        if (item.column() == COL_NAME and self._drag_start_row >= 0
                and row != self._drag_start_row):
            order = list(range(self._model.rowCount()))
            self.label_reordered.emit(order)
            self._drag_start_row = -1

    def _on_selection_changed(self, *_args) -> None:
        row = self.selected_row()
        if row >= 0:
            self.label_selected.emit(row)

    def _delete_selected(self) -> None:
        row = self.selected_row()
        if row >= 0:
            self.remove_label(row)

    def _duplicate_selected(self) -> None:
        row = self.selected_row()
        if row < 0:
            return
        name = self.label_name(row) + " (copy)"
        new_row = self.add_label(name, self.label_color(row),
                                 source="Duplicate", emit=False)
        self.label_duplicate_requested.emit(row)
        self.select_row(new_row)

    def _export_selected(self) -> None:
        row = self.selected_row()
        if row >= 0:
            self.label_export_requested.emit(row)

    # ------------------------------------------------------------------ #
    # Context menu
    # ------------------------------------------------------------------ #
    def _show_context_menu(self, position: QPoint) -> None:
        index = self._view.indexAt(position)
        if not index.isValid():
            return
        row = index.row()
        self.select_row(row)

        menu = QMenu(self)
        act_rename = QAction("Rename", menu)
        act_color = QAction("Change Colour...", menu)
        act_fill = QAction("Fill Holes", menu)

        bool_menu = menu.addMenu("Boolean Operation")
        act_union = QAction("Union", bool_menu)
        act_intersect = QAction("Intersect", bool_menu)
        act_subtract = QAction("Subtract", bool_menu)

        act_rebuild = QAction("Rebuild 3D", menu)
        act_stats = QAction("Statistics", menu)
        menu.addSeparator()
        act_delete = QAction("Delete", menu)

        act_rename.triggered.connect(lambda: self.label_rename_requested.emit(row))
        act_color.triggered.connect(lambda: self._on_clicked(
            self._model.index(row, COL_COLOR)))
        act_fill.triggered.connect(
            lambda: self.label_fill_holes_requested.emit(row))
        act_union.triggered.connect(
            lambda: self.label_boolean_requested.emit(row, "union"))
        act_intersect.triggered.connect(
            lambda: self.label_boolean_requested.emit(row, "intersect"))
        act_subtract.triggered.connect(
            lambda: self.label_boolean_requested.emit(row, "subtract"))
        act_rebuild.triggered.connect(
            lambda: self.label_rebuild_requested.emit(row))
        act_stats.triggered.connect(
            lambda: self.label_statistics_requested.emit(row))
        act_delete.triggered.connect(lambda: self.remove_label(row))

        menu.exec_(self._view.viewport().mapToGlobal(position))
