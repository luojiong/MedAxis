"""
Model Zoo — browse, import and download AI models.

Models are registered as metadata entries; archives can be downloaded into
the local cache or imported from disk.  Actual inference runs through the
external AI service clients (plugins/api_clients).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .model_downloader import MODEL_CATALOG, ModelDownloader


class ModelZooWidget(QWidget):
    """Panel to browse the model catalog, import and download models."""

    model_imported = Signal(str)  # model id / path

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._downloader = ModelDownloader()
        self._entries = list(MODEL_CATALOG)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("AI Model Zoo", self))

        self._list = QListWidget(self)
        self._refresh_list()
        layout.addWidget(self._list, 1)

        self._progress = QProgressBar(self)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        row = QHBoxLayout()
        self._import_btn = QPushButton("Import from disk…", self)
        self._import_btn.clicked.connect(self._import_model)
        self._details_btn = QPushButton("Details", self)
        self._details_btn.clicked.connect(self._show_details)
        row.addWidget(self._import_btn)
        row.addWidget(self._details_btn)
        row.addStretch(1)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    def _refresh_list(self) -> None:
        self._list.clear()
        cached = {p.stem for p in self._downloader.list_models()}
        for entry in self._entries:
            marker = "✓" if entry["id"] in cached else " "
            item = QListWidgetItem(f"{marker} {entry['name']}  ({entry['modality']})")
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            self._list.addItem(item)

    def _selected(self) -> Optional[dict]:
        item = self._list.currentItem()
        if item is None:
            return None
        model_id = item.data(Qt.ItemDataRole.UserRole)
        return next((e for e in self._entries if e["id"] == model_id), None)

    def _import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Model Archive", "",
            "Archives (*.zip *.tar.gz *.onnx);;All Files (*)")
        if not path:
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        try:
            self._downloader.extract_archive(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
        finally:
            self._progress.setVisible(False)
        self.model_imported.emit(path)
        self._refresh_list()

    def _show_details(self) -> None:
        entry = self._selected()
        if entry is None:
            return
        QMessageBox.information(
            self, entry["name"],
            f"Modality: {entry['modality']}\n"
            f"Classes: {entry['num_classes']}\n"
            f"Source: {entry['source']}\n\n{entry['description']}")

    def register_model(self, entry: dict) -> None:
        """Add a custom model entry to the zoo."""
        if entry.get("id") not in [e["id"] for e in self._entries]:
            self._entries.append(entry)
            self._refresh_list()
