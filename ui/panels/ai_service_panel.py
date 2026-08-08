"""AI service management panel for MedAxis.

:class:`AIServicePanel` manages remote AI segmentation services:

* a list of registered services (name, URL, live status indicator);
* Add / Edit / Delete controls;
* a connection-test button;
* a model browser populated from the selected service;
* a *Run AI segmentation* trigger;
* a job monitor showing queued/running jobs with progress.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

STATUS_COLORS = {
    "online": QColor(60, 200, 90),
    "offline": QColor(220, 70, 60),
    "testing": QColor(230, 180, 40),
    "unknown": QColor(150, 150, 150),
}


class _ServiceDialog(QDialog):
    """Small modal form used for adding / editing a service entry."""

    def __init__(self, name: str = "", url: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Service")
        form = QFormLayout(self)
        self._name = QLineEdit(name, self)
        self._url = QLineEdit(url, self)
        self._url.setPlaceholderText("http://host:port")
        form.addRow("Name:", self._name)
        form.addRow("URL:", self._url)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return self._name.text().strip(), self._url.text().strip()


class AIServicePanel(QWidget):
    """Registration and control panel for remote AI services.

    Signals
    -------
    service_added(dict)
        New service descriptor ``{"name": ..., "url": ...}``.
    service_removed(str)
        Name of the removed service.
    service_edited(dict)
        Updated descriptor after *Edit*.
    connection_test_requested(dict)
        Descriptor of the service to test.
    segmentation_requested(dict)
        ``{"service": name, "model": model_id}`` when *Run* is clicked.
    job_cancel_requested(str)
        Job ID the user wants cancelled.
    """

    service_added = Signal(dict)
    service_removed = Signal(str)
    service_edited = Signal(dict)
    connection_test_requested = Signal(dict)
    segmentation_requested = Signal(dict)
    job_cancel_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AIServicePanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -------------------------------------------------------------- #
        # Service list
        # -------------------------------------------------------------- #
        services_group = QGroupBox("Services", self)
        services_layout = QVBoxLayout(services_group)

        self._service_tree = QTreeWidget(services_group)
        self._service_tree.setColumnCount(3)
        self._service_tree.setHeaderLabels(["Name", "URL", "Status"])
        self._service_tree.setRootIsDecorated(False)
        self._service_tree.itemSelectionChanged.connect(
            self._on_service_selected)
        header = self._service_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        services_layout.addWidget(self._service_tree, 1)

        buttons = QHBoxLayout()
        self._btn_add = QPushButton("Add", services_group)
        self._btn_edit = QPushButton("Edit", services_group)
        self._btn_delete = QPushButton("Delete", services_group)
        self._btn_test = QPushButton("Test Connection", services_group)
        self._btn_add.clicked.connect(self._add_service)
        self._btn_edit.clicked.connect(self._edit_service)
        self._btn_delete.clicked.connect(self._delete_service)
        self._btn_test.clicked.connect(self._test_connection)
        for btn in (self._btn_add, self._btn_edit, self._btn_delete,
                    self._btn_test):
            buttons.addWidget(btn)
        services_layout.addLayout(buttons)
        layout.addWidget(services_group, 2)

        # -------------------------------------------------------------- #
        # Model browser
        # -------------------------------------------------------------- #
        models_group = QGroupBox("Models", self)
        models_layout = QVBoxLayout(models_group)
        self._model_list = QListWidget(models_group)
        models_layout.addWidget(self._model_list, 1)
        self._btn_run = QPushButton("Run AI Segmentation", models_group)
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run_segmentation)
        models_layout.addWidget(self._btn_run)
        layout.addWidget(models_group, 2)

        # -------------------------------------------------------------- #
        # Job monitor
        # -------------------------------------------------------------- #
        jobs_group = QGroupBox("Jobs", self)
        jobs_layout = QVBoxLayout(jobs_group)
        self._job_tree = QTreeWidget(jobs_group)
        self._job_tree.setColumnCount(4)
        self._job_tree.setHeaderLabels(["Job", "Model", "Status", "Progress"])
        self._job_tree.setRootIsDecorated(False)
        jobs_layout.addWidget(self._job_tree, 1)
        self._btn_cancel_job = QPushButton("Cancel Job", jobs_group)
        self._btn_cancel_job.clicked.connect(self._cancel_job)
        jobs_layout.addWidget(self._btn_cancel_job)
        layout.addWidget(jobs_group, 1)

    # ================================================================== #
    # Service management
    # ================================================================== #
    def add_service(self, name: str, url: str,
                    status: str = "unknown") -> None:
        item = QTreeWidgetItem([name, url, status])
        self._paint_status(item, status)
        self._service_tree.addTopLevelItem(item)

    def _selected_service_item(self) -> Optional[QTreeWidgetItem]:
        items = self._service_tree.selectedItems()
        return items[0] if items else None

    def _add_service(self) -> None:
        dialog = _ServiceDialog(parent=self)
        if dialog.exec() == QDialog.Accepted:
            name, url = dialog.values()
            if name and url:
                self.add_service(name, url)
                self.service_added.emit({"name": name, "url": url})

    def _edit_service(self) -> None:
        item = self._selected_service_item()
        if item is None:
            return
        dialog = _ServiceDialog(item.text(0), item.text(1), self)
        if dialog.exec() == QDialog.Accepted:
            name, url = dialog.values()
            if name and url:
                item.setText(0, name)
                item.setText(1, url)
                self.service_edited.emit({"name": name, "url": url})

    def _delete_service(self) -> None:
        item = self._selected_service_item()
        if item is None:
            return
        name = item.text(0)
        index = self._service_tree.indexOfTopLevelItem(item)
        self._service_tree.takeTopLevelItem(index)
        self.service_removed.emit(name)

    def _test_connection(self) -> None:
        item = self._selected_service_item()
        if item is None:
            return
        self.set_service_status(item.text(0), "testing")
        self.connection_test_requested.emit(
            {"name": item.text(0), "url": item.text(1)})

    def set_service_status(self, name: str, status: str,
                           message: str = "") -> None:
        """Update the status indicator for a service (called by the
        controller after a connection test)."""
        for row in range(self._service_tree.topLevelItemCount()):
            item = self._service_tree.topLevelItem(row)
            if item.text(0) == name:
                label = status if not message else f"{status}: {message}"
                item.setText(2, label)
                self._paint_status(item, status)
                return

    def _paint_status(self, item: QTreeWidgetItem, status: str) -> None:
        color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
        item.setForeground(2, QBrush(color))

    def _on_service_selected(self) -> None:
        item = self._selected_service_item()
        self._btn_run.setEnabled(item is not None
                                 and self._model_list.count() > 0)

    # ================================================================== #
    # Model browser
    # ================================================================== #
    def set_models(self, models: List[dict]) -> None:
        """Populate the model list. Each entry should look like
        ``{"id": str, "name": str, "description": str}``."""
        self._model_list.clear()
        for model in models:
            item = QListWidgetItem(model.get("name", model.get("id", "?")))
            item.setData(Qt.UserRole, model)
            item.setToolTip(model.get("description", ""))
            self._model_list.addItem(item)
        self._btn_run.setEnabled(self._selected_service_item() is not None
                                 and self._model_list.count() > 0)

    def selected_model(self) -> Optional[dict]:
        item = self._model_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _run_segmentation(self) -> None:
        service_item = self._selected_service_item()
        model = self.selected_model()
        if service_item is None or model is None:
            return
        self.segmentation_requested.emit({
            "service": service_item.text(0),
            "url": service_item.text(1),
            "model": model.get("id", model.get("name")),
        })

    # ================================================================== #
    # Job monitor
    # ================================================================== #
    def add_job(self, job_id: str, model: str,
                status: str = "queued") -> int:
        item = QTreeWidgetItem([job_id, model, status, "0%"])
        self._job_tree.addTopLevelItem(item)
        return self._job_tree.topLevelItemCount() - 1

    def update_job(self, row: int, status: Optional[str] = None,
                   progress: Optional[int] = None) -> None:
        item = self._job_tree.topLevelItem(row)
        if item is None:
            return
        if status is not None:
            item.setText(2, status)
        if progress is not None:
            item.setText(3, f"{int(progress)}%")

    def remove_job(self, row: int) -> None:
        self._job_tree.takeTopLevelItem(row)

    def _cancel_job(self) -> None:
        item = self._job_tree.currentItem()
        if item is not None:
            self.job_cancel_requested.emit(item.text(0))
