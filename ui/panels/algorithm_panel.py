"""Algorithm catalogue & parameter panel for MedAxis.

:class:`AlgorithmPanel` presents the available algorithms as a category
tree (Filtering / Morphology / Segmentation / Registration / ...),
generates a parameter form dynamically from an algorithm's schema, and
offers Run / Preview controls with a progress bar and a queue list for
running and pending jobs.

Schema format
-------------
An algorithm schema is a list of parameter dicts::

    [
        {"name": "sigma", "label": "Sigma", "type": "float",
         "default": 1.0, "min": 0.1, "max": 10.0},
        {"name": "iterations", "label": "Iterations", "type": "int",
         "default": 5, "min": 1, "max": 100},
        {"name": "method", "label": "Method", "type": "choice",
         "options": ["Fast", "Accurate"], "default": "Fast"},
        {"name": "use_gpu", "label": "Use GPU", "type": "bool",
         "default": True},
    ]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


class AlgorithmPanel(QWidget):
    """Algorithm browser with dynamic parameter form and job queue.

    Signals
    -------
    algorithm_selected(str, str)
        Emitted with ``(category, algorithm_name)`` on selection.
    algorithm_run_requested(dict)
        Emitted when *Run* is clicked. Payload::

            {"category": str, "algorithm": str, "params": dict,
             "preview": bool}
    algorithm_cancel_requested(int)
        Emitted with the queue row of the job to cancel.
    """

    algorithm_selected = Signal(str, str)
    algorithm_run_requested = Signal(dict)
    algorithm_cancel_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AlgorithmPanel")

        self._current_category: str = ""
        self._current_algorithm: str = ""
        self._schema: List[dict] = []
        self._param_widgets: Dict[str, QWidget] = {}
        self._catalog: Dict[str, List[str]] = {}

        splitter = QSplitter(Qt.Vertical, self)

        # -------------------------------------------------------------- #
        # Category tree (top)
        # -------------------------------------------------------------- #
        self._tree = QTreeWidget(splitter)
        self._tree.setColumnCount(1)
        self._tree.setHeaderLabel("Algorithms")
        self._tree.itemClicked.connect(self._on_item_clicked)
        splitter.addWidget(self._tree)

        # -------------------------------------------------------------- #
        # Parameter form (middle)
        # -------------------------------------------------------------- #
        form_group = QGroupBox("Parameters", splitter)
        form_layout = QVBoxLayout(form_group)
        self._form = QFormLayout()
        self._form_widget = QWidget(form_group)
        self._form_widget.setLayout(self._form)
        form_layout.addWidget(self._form_widget, 1)

        # Run controls.
        controls = QHBoxLayout()
        self._preview_check = QCheckBox("Preview", form_group)
        self._run_button = QPushButton("Run", form_group)
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run)
        self._cancel_button = QPushButton("Cancel", form_group)
        self._cancel_button.clicked.connect(self._on_cancel)
        controls.addWidget(self._preview_check)
        controls.addWidget(self._run_button)
        controls.addWidget(self._cancel_button)
        form_layout.addLayout(controls)

        self._progress = QProgressBar(form_group)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        form_layout.addWidget(self._progress)

        splitter.addWidget(form_group)

        # -------------------------------------------------------------- #
        # Queue list (bottom)
        # -------------------------------------------------------------- #
        queue_group = QGroupBox("Queue", splitter)
        queue_layout = QVBoxLayout(queue_group)
        self._queue = QTreeWidget(queue_group)
        self._queue.setColumnCount(3)
        self._queue.setHeaderLabels(["Job", "Status", "Progress"])
        self._queue.setRootIsDecorated(False)
        queue_layout.addWidget(self._queue)
        splitter.addWidget(queue_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ================================================================== #
    # Catalogue
    # ================================================================== #
    def set_catalog(self, catalog: Dict[str, List[str]]) -> None:
        """Populate the category tree, e.g.
        ``{"Filtering": ["Gaussian", "Median"], ...}``."""
        self._catalog = catalog
        self._tree.clear()
        for category, algorithms in catalog.items():
            cat_item = QTreeWidgetItem([category])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            self._tree.addTopLevelItem(cat_item)
            for algorithm in algorithms:
                alg_item = QTreeWidgetItem([algorithm])
                alg_item.setData(0, Qt.UserRole, category)
                cat_item.addChild(alg_item)
            cat_item.setExpanded(True)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        category = item.data(0, Qt.UserRole)
        if category is None:
            return  # category header clicked
        self._current_category = category
        self._current_algorithm = item.text(0)
        self._run_button.setEnabled(True)
        self.algorithm_selected.emit(category, self._current_algorithm)
        # Ask the controller (if any) for the schema.
        provider = getattr(self, "_schema_provider", None)
        if callable(provider):
            schema = provider(category, self._current_algorithm)
            if schema:
                self.load_schema(schema)

    def set_schema_provider(self, provider) -> None:
        """Register ``provider(category, name) -> schema list``."""
        self._schema_provider = provider

    # ================================================================== #
    # Dynamic parameter form
    # ================================================================== #
    def load_schema(self, schema: List[dict]) -> None:
        """(Re)build the parameter form from *schema*."""
        self._schema = list(schema)
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._param_widgets.clear()

        for spec in self._schema:
            name = spec["name"]
            label = spec.get("label", name)
            ptype = spec.get("type", "str")
            default = spec.get("default")

            widget: QWidget
            if ptype == "int":
                spin = QSpinBox(self._form_widget)
                spin.setRange(int(spec.get("min", -10 ** 9)),
                              int(spec.get("max", 10 ** 9)))
                if default is not None:
                    spin.setValue(int(default))
                widget = spin
            elif ptype == "float":
                spin = QDoubleSpinBox(self._form_widget)
                spin.setRange(float(spec.get("min", -1e12)),
                              float(spec.get("max", 1e12)))
                spin.setDecimals(int(spec.get("decimals", 3)))
                if default is not None:
                    spin.setValue(float(default))
                widget = spin
            elif ptype == "bool":
                check = QCheckBox(self._form_widget)
                check.setChecked(bool(default))
                widget = check
            elif ptype == "choice":
                combo = QComboBox(self._form_widget)
                combo.addItems([str(o) for o in spec.get("options", [])])
                if default is not None:
                    idx = combo.findText(str(default))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                widget = combo
            else:  # plain string
                from PySide6.QtWidgets import QLineEdit
                edit = QLineEdit(self._form_widget)
                if default is not None:
                    edit.setText(str(default))
                widget = edit

            self._form.addRow(label + ":", widget)
            self._param_widgets[name] = widget

    def collect_parameters(self) -> Dict[str, Any]:
        """Read the current values of every form field."""
        from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, \
            QLineEdit, QSpinBox

        values: Dict[str, Any] = {}
        for spec in self._schema:
            name = spec["name"]
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text()
        return values

    # ================================================================== #
    # Job queue
    # ================================================================== #
    def add_job(self, name: str, status: str = "pending") -> int:
        """Append a job row; returns the row index."""
        item = QTreeWidgetItem([name, status, "0%"])
        self._queue.addTopLevelItem(item)
        return self._queue.topLevelItemCount() - 1

    def update_job(self, row: int, status: Optional[str] = None,
                   progress: Optional[int] = None) -> None:
        item = self._queue.topLevelItem(row)
        if item is None:
            return
        if status is not None:
            item.setText(1, status)
        if progress is not None:
            item.setText(2, f"{int(progress)}%")

    def remove_job(self, row: int) -> None:
        self._queue.takeTopLevelItem(row)

    def clear_finished_jobs(self) -> None:
        for row in range(self._queue.topLevelItemCount() - 1, -1, -1):
            status = self._queue.topLevelItem(row).text(1).lower()
            if status in ("done", "finished", "failed", "cancelled"):
                self._queue.takeTopLevelItem(row)

    def set_progress(self, value: int, visible: bool = True) -> None:
        self._progress.setValue(max(0, min(int(value), 100)))
        self._progress.setVisible(visible)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_run(self) -> None:
        if not self._current_algorithm:
            return
        payload = {
            "category": self._current_category,
            "algorithm": self._current_algorithm,
            "params": self.collect_parameters(),
            "preview": self._preview_check.isChecked(),
        }
        self.algorithm_run_requested.emit(payload)

    def _on_cancel(self) -> None:
        item = self._queue.currentItem()
        if item is None:
            return
        row = self._queue.indexOfTopLevelItem(item)
        self.algorithm_cancel_requested.emit(row)
