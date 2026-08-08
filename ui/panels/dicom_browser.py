"""DICOM browser dock for MedAxis.

:class:`DICOMBrowser` is a :class:`QDockWidget` presenting the DICOM
database as a patient -> study -> series hierarchy in a tree view, with:

* drag & drop of DICOM files and folders (parsed with *pydicom* when
  available);
* right-click context menu (Load / Unload / Properties / Export);
* a search / filter bar;
* hover preview of series information (thumbnail when pydicom is present).
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import QModelIndex, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QAction, QImage, QStandardItem, \
    QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QDockWidget, QHBoxLayout, QHeaderView, QLineEdit, QMenu, QTreeView, QVBoxLayout, QWidget,
)

# Roles storing DICOM identifiers on tree items.
ROLE_UID = Qt.UserRole + 1
ROLE_LEVEL = Qt.UserRole + 2   # "patient" | "study" | "series"

try:  # optional dependency
    import pydicom  # type: ignore
    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    HAS_PYDICOM = False


class DICOMBrowser(QDockWidget):
    """Patient/study/series browser with drag & drop support.

    Signals
    -------
    series_selected(str)
        Emitted with the series instance UID when a series row is selected.
    series_load_requested(str)
        Emitted when the user asks to load a series (double click or the
        *Load* context action).
    files_dropped(list)
        Emitted with the list of dropped file paths when pydicom is not
        available (the application layer can index them itself).
    """

    series_selected = Signal(str)
    series_load_requested = Signal(str)
    files_dropped = Signal(list)

    def __init__(self, title: str = "DICOM Browser",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("DICOMBrowser")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(260)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search / filter bar.
        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit(container)
        self._filter_edit.setPlaceholderText("Search patient / study / "
                                             "series...")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit)
        layout.addLayout(filter_row)

        # Hierarchy tree.
        self._model = QStandardItemModel(container)
        self._model.setHorizontalHeaderLabels(["Name", "Info"])
        self._tree = QTreeView(container)
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(False)
        self._tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.setMouseTracking(True)
        self._tree.entered.connect(self._on_hover)
        self._tree.clicked.connect(self._on_clicked)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self._tree, 1)

        self.setWidget(container)

        # Drag & drop.
        self.setAcceptDrops(True)

    # ================================================================== #
    # Public API for populating the tree
    # ================================================================== #
    def add_series(self, patient_name: str, study_description: str,
                   series_description: str, series_uid: str,
                   patient_id: str = "", study_uid: str = "",
                   num_instances: int = 0) -> None:
        """Insert (or merge) a series under patient -> study."""
        patient_item = self._find_or_create_root(
            f"{patient_name} ({patient_id})" if patient_id else patient_name,
            "patient", patient_id or patient_name)

        study_label = study_description or "Study"
        study_item = self._find_child(patient_item, study_label, "study",
                                      study_uid)

        series_label = series_description or "Series"
        info = f"{num_instances} img" if num_instances else ""
        self._find_child(study_item, series_label, "series", series_uid, info)
        self._tree.expand(self._model.indexFromItem(patient_item))

    def clear(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    # ------------------------------------------------------------------ #
    def _find_or_create_root(self, text: str, level: str,
                             uid: str) -> QStandardItem:
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item.data(ROLE_LEVEL) == level and item.data(ROLE_UID) == uid:
                return item
        name_item = QStandardItem(text)
        name_item.setData(level, ROLE_LEVEL)
        name_item.setData(uid, ROLE_UID)
        name_item.setEditable(False)
        self._model.appendRow([name_item, QStandardItem("")])
        return name_item

    def _find_child(self, parent: QStandardItem, text: str, level: str,
                    uid: str, info: str = "") -> QStandardItem:
        for row in range(parent.rowCount()):
            item = parent.child(row, 0)
            if item.data(ROLE_LEVEL) == level and item.data(ROLE_UID) == uid:
                return item
        name_item = QStandardItem(text)
        name_item.setData(level, ROLE_LEVEL)
        name_item.setData(uid, ROLE_UID)
        name_item.setEditable(False)
        info_item = QStandardItem(info)
        info_item.setEditable(False)
        parent.appendRow([name_item, info_item])
        return name_item

    # ================================================================== #
    # Selection / signals
    # ================================================================== #
    def _on_clicked(self, index: QModelIndex) -> None:
        item = self._model.itemFromIndex(index)
        if item is None:
            return
        if item.data(ROLE_LEVEL) == "series":
            self.series_selected.emit(item.data(ROLE_UID))

    def _on_double_clicked(self, index: QModelIndex) -> None:
        item = self._model.itemFromIndex(index)
        if item is None:
            return
        if item.data(ROLE_LEVEL) == "series":
            self.series_load_requested.emit(item.data(ROLE_UID))

    # ================================================================== #
    # Hover preview
    # ================================================================== #
    def _on_hover(self, index: QModelIndex) -> None:
        item = self._model.itemFromIndex(index)
        if item is None:
            return
        level = item.data(ROLE_LEVEL)
        uid = item.data(ROLE_UID) or ""
        tip = f"<b>{item.text()}</b><br/>{level}: {uid}"
        if level == "series" and HAS_PYDICOM:
            thumb = self._series_thumbnail(uid)
            if thumb is not None:
                tip += f'<br/><img src="{thumb}"/>'
        self._tree.setToolTip(tip)

    def _series_thumbnail(self, series_uid: str) -> Optional[str]:
        """Create a thumbnail PNG of the first slice of a series if the
        source files are known; returns a temp file path or None."""
        source = getattr(self, "_series_sources", {}).get(series_uid)
        if not source or not HAS_PYDICOM:
            return None
        try:  # pragma: no cover - depends on actual DICOM data
            ds = pydicom.dcmread(source, stop_before_pixels=False)
            arr = ds.pixel_array
            arr = ((arr - arr.min()) / max(arr.ptp(), 1) * 255).astype("ubyte")
            h, w = arr.shape
            image = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            path = os.path.join(os.environ.get("TEMP", "/tmp"),
                                f"medaxis_thumb_{abs(hash(series_uid))}.png")
            image.scaled(QSize(128, 128), Qt.KeepAspectRatio).save(path)
            return path
        except Exception:
            return None

    # ================================================================== #
    # Context menu
    # ================================================================== #
    def _show_context_menu(self, position: QPoint) -> None:
        index = self._tree.indexAt(position)
        if not index.isValid():
            return
        item = self._model.itemFromIndex(index)
        if item is None or item.data(ROLE_LEVEL) != "series":
            return
        uid = item.data(ROLE_UID)

        menu = QMenu(self)
        act_load = QAction("Load", menu)
        act_unload = QAction("Unload", menu)
        act_props = QAction("Properties...", menu)
        act_export = QAction("Export...", menu)

        act_load.triggered.connect(lambda: self.series_load_requested.emit(uid))
        act_unload.triggered.connect(lambda: self._series_action("unload", uid))
        act_props.triggered.connect(lambda: self._series_action("properties", uid))
        act_export.triggered.connect(lambda: self._series_action("export", uid))

        menu.addAction(act_load)
        menu.addAction(act_unload)
        menu.addSeparator()
        menu.addAction(act_props)
        menu.addAction(act_export)
        menu.exec_(self._tree.viewport().mapToGlobal(position))

    def _series_action(self, action: str, uid: str) -> None:
        """Placeholder hook for unload / properties / export — the
        application controller can override or connect to these."""
        handler = getattr(self, f"on_series_{action}", None)
        if callable(handler):
            handler(uid)

    # ================================================================== #
    # Filter
    # ================================================================== #
    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()

        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            matches = needle in item.text().lower() or not needle
            child_matches = False
            for r2 in range(item.rowCount()):
                child = item.child(r2, 0)
                if child and self._subtree_matches(child, needle):
                    child_matches = True
            visible = matches or child_matches
            self._tree.setRowHidden(row, QModelIndex(), not visible)
            if visible and needle:
                self._tree.expand(self._model.indexFromItem(item))
            for r2 in range(item.rowCount()):
                child = item.child(r2, 0)
                if child is not None:
                    child_visible = self._subtree_matches(child, needle) \
                        or not needle
                    self._tree.setRowHidden(r2,
                                            self._model.indexFromItem(item),
                                            not child_visible)

    def _subtree_matches(self, item: QStandardItem, needle: str) -> bool:
        if not needle:
            return True
        if needle in item.text().lower():
            return True
        for row in range(item.rowCount()):
            child = item.child(row, 0)
            if child is not None and self._subtree_matches(child, needle):
                return True
        return False

    # ================================================================== #
    # Drag & drop of DICOM files / folders
    # ================================================================== #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths: List[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            if os.path.isdir(local):
                for root, _dirs, files in os.walk(local):
                    for fname in files:
                        if self._looks_like_dicom(fname):
                            paths.append(os.path.join(root, fname))
            elif self._looks_like_dicom(local):
                paths.append(local)

        if not paths:
            return
        event.acceptProposedAction()

        if not HAS_PYDICOM:
            self.files_dropped.emit(paths)
            return

        self._index_files(paths)

    @staticmethod
    def _looks_like_dicom(fname: str) -> bool:
        lower = fname.lower()
        return (lower.endswith((".dcm", ".dicom", ".ima"))
                or "." not in os.path.basename(lower))

    def _index_files(self, paths: List[str]) -> None:  # pragma: no cover
        if not hasattr(self, "_series_sources"):
            self._series_sources = {}
        for path in paths:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                continue
            patient = str(getattr(ds, "PatientName", "Unknown"))
            patient_id = str(getattr(ds, "PatientID", ""))
            study_desc = str(getattr(ds, "StudyDescription", "")
                             or getattr(ds, "StudyDate", "Study"))
            series_desc = str(getattr(ds, "SeriesDescription", "Series"))
            series_uid = str(getattr(ds, "SeriesInstanceUID", path))
            study_uid = str(getattr(ds, "StudyInstanceUID", ""))
            self._series_sources[series_uid] = path
            self.add_series(patient, study_desc, series_desc, series_uid,
                            patient_id=patient_id, study_uid=study_uid)
