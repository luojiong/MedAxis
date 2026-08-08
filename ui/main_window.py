"""Main application window for the MedAxis medical imaging platform.

:class:`MainWindow` assembles the complete UI:

* menu bar (File / Edit / View / Image / Measure / Segment / Analyze /
  Tools / Help);
* the main :class:`MedAxisToolBar`;
* a central horizontal splitter — 72 % viewports
  (:class:`ViewContainer`) and 28 % right-side property panel
  (:class:`PropertyPanel` with Masks / Objects / Export tabs);
* a status bar showing coordinates, zoom level, a progress bar and GPU
  memory usage;
* dock widgets: DICOM browser (left), script console (bottom) and an AI
  services panel (right).

The window is controller-agnostic: call :meth:`MainWindow.attach_controller`
with an ``AppController`` to wire every action and signal.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QSplitter,
    QWidget,
)

from .panels.ai_service_panel import AIServicePanel
from .panels.algorithm_panel import AlgorithmPanel
from .panels.dicom_browser import DICOMBrowser
from .panels.property_panel import PropertyPanel
from .panels.toolbar import MedAxisToolBar
from scripts.console import ScriptConsole
from scripts.editor import ScriptEditor
from .panels.view_container import ViewContainer
from .widgets.label_editor import LabelEditor


class ScriptConsoleWidget(ScriptConsole):
    """Script REPL docked at the bottom of the main window."""

    pass


class MainWindow(QMainWindow):
    """Top-level MedAxis application window."""

    RECENT_FILES_KEY = "recentFiles"
    MAX_RECENT_FILES = 10

    def __init__(self, controller=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MedAxis — Medical Imaging Platform")
        self.resize(1600, 950)

        self._settings = QSettings("MedAxis", "MedAxis")
        self._dirty = False
        self._current_file: Optional[str] = None
        self.controller = None
        self._recent_menu = None
        self._recent_actions: List[QAction] = []

        # ------------------------------------------------------------- #
        # Build UI
        # ------------------------------------------------------------- #
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central_widget()
        self._build_status_bar()
        self._build_docks()
        self._update_recent_menu()

        if controller is not None:
            self.attach_controller(controller)

    # ================================================================== #
    # Actions & menus
    # ================================================================== #
    def _make_action(self, text: str, slot=None,
                     shortcut: Optional[str] = None,
                     checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if checkable:
            action.setCheckable(True)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        a = {}
        # File
        a["new"] = self._make_action("&New Session", self._new_session, "Ctrl+N")
        a["open"] = self._make_action("&Open...", self._open_file, "Ctrl+O")
        a["save"] = self._make_action("&Save", self._save_file, "Ctrl+S")
        a["save_as"] = self._make_action("Save &As...", self._save_file_as,
                                         "Ctrl+Shift+S")
        a["export"] = self._make_action("&Export...", self._export_requested)
        a["quit"] = self._make_action("&Quit", self.close, "Ctrl+Q")
        # Edit
        a["undo"] = self._make_action("&Undo", self._undo, "Ctrl+Z")
        a["redo"] = self._make_action("&Redo", self._redo, "Ctrl+Y")
        a["settings"] = self._make_action("&Settings...", self._open_settings)
        # Image
        a["window_level_reset"] = self._make_action(
            "Reset &Window/Level", self._reset_window_level, "Ctrl+L")
        # Measure
        a["measure_distance"] = self._make_action(
            "&Distance", lambda: self._set_tool("distance"), "D")
        a["measure_angle"] = self._make_action(
            "&Angle", lambda: self._set_tool("angle"), "A")
        a["measure_roi"] = self._make_action(
            "&ROI Statistics", self._roi_statistics)
        # Segment
        a["seg_threshold"] = self._make_action(
            "&Threshold...", self._segment_threshold)
        a["seg_region_grow"] = self._make_action(
            "&Region Growing...", self._segment_region_grow)
        a["seg_watershed"] = self._make_action(
            "&Watershed...", self._segment_watershed)
        a["seg_paint"] = self._make_action(
            "&Paint", lambda: self._set_tool("paint"), "P")
        a["seg_erase"] = self._make_action(
            "&Erase", lambda: self._set_tool("erase"), "E")
        a["seg_fill"] = self._make_action(
            "&Fill", lambda: self._set_tool("fill"), "F")
        # Analyze
        a["histogram"] = self._make_action("&Histogram", self._show_histogram)
        a["radiomics"] = self._make_action("&Radiomics...", self._show_radiomics)
        # Help
        a["about"] = self._make_action("&About MedAxis", self._show_about)

        self.actions = a

    def _build_menus(self) -> None:
        a = self.actions
        bar = self.menuBar()

        # --- File ---
        file_menu = bar.addMenu("&File")
        file_menu.addAction(a["new"])
        file_menu.addAction(a["open"])
        file_menu.addSeparator()
        file_menu.addAction(a["save"])
        file_menu.addAction(a["save_as"])
        self._recent_menu = file_menu.addMenu("Open &Recent")
        file_menu.addSeparator()
        file_menu.addAction(a["export"])
        file_menu.addSeparator()
        file_menu.addAction(a["quit"])

        # --- Edit ---
        edit_menu = bar.addMenu("&Edit")
        edit_menu.addAction(a["undo"])
        edit_menu.addAction(a["redo"])
        edit_menu.addSeparator()
        edit_menu.addAction(a["settings"])

        # --- View ---
        view_menu = bar.addMenu("&View")
        layouts_menu = view_menu.addMenu("&Layouts")
        self._layout_actions = {}
        for mode, label in (("1up", "1-up"), ("2up_h", "2-up Horizontal"),
                            ("2up_v", "2-up Vertical"), ("4up", "4-up"),
                            ("1plus3", "1 + 3")):
            act = self._make_action(
                label, lambda _=False, m=mode: self._set_layout(m))
            act.setCheckable(True)
            if mode == "4up":
                act.setChecked(True)
            layouts_menu.addAction(act)
            self._layout_actions[mode] = act
        view_menu.addSeparator()
        compare_action = self._make_action(
            "&Comparison View…", None, None)
        compare_action.triggered.connect(self._open_comparison_view)
        view_menu.addAction(compare_action)
        self._compare_action = compare_action
        view_menu.addSeparator()
        # Dock toggles are added after the docks exist (see _build_docks).

        # --- Image ---
        image_menu = bar.addMenu("&Image")
        image_menu.addAction(a["window_level_reset"])
        presets_menu = image_menu.addMenu("Window/Level &Presets")
        for preset, wl in (("CT Bone", (1500, 300)),
                           ("CT Soft Tissue", (400, 40)),
                           ("CT Lung", (1500, -600)),
                           ("CT Liver", (150, 30)),
                           ("Brain", (80, 40))):
            act = self._make_action(
                preset, lambda _=False, w=wl: self._apply_window_level(*w))
            presets_menu.addAction(act)

        # --- Measure ---
        measure_menu = bar.addMenu("&Measure")
        measure_menu.addAction(a["measure_distance"])
        measure_menu.addAction(a["measure_angle"])
        measure_menu.addAction(a["measure_roi"])

        # --- Segment ---
        segment_menu = bar.addMenu("&Segment")
        segment_menu.addAction(a["seg_threshold"])
        segment_menu.addAction(a["seg_region_grow"])
        segment_menu.addAction(a["seg_watershed"])
        segment_menu.addSeparator()
        segment_menu.addAction(a["seg_paint"])
        segment_menu.addAction(a["seg_erase"])
        segment_menu.addAction(a["seg_fill"])

        # --- Analyze ---
        analyze_menu = bar.addMenu("A&nalyze")
        analyze_menu.addAction(a["histogram"])
        analyze_menu.addAction(a["radiomics"])

        # --- Tools ---
        tools_menu = bar.addMenu("&Tools")
        self._tools_console_action = self._make_action(
            "&Script Console", None, "Ctrl+`", checkable=True)
        self._tools_ai_action = self._make_action(
            "&AI Services", None, None, checkable=True)
        tools_menu.addAction(self._tools_console_action)
        tools_menu.addAction(self._tools_ai_action)

        # --- Help ---
        help_menu = bar.addMenu("&Help")
        help_menu.addAction(a["about"])

    def _build_toolbar(self) -> None:
        self.toolbar = MedAxisToolBar(self)
        self.addToolBar(self.toolbar)
        self.toolbar.tool_changed.connect(self._set_tool)

    # ================================================================== #
    # Central widget: viewports (72%) | property panel (28%)
    # ================================================================== #
    def _build_central_widget(self) -> None:
        self.view_container = ViewContainer(self)
        self.property_panel = PropertyPanel(self)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.view_container)
        splitter.addWidget(self.property_panel)
        splitter.setStretchFactor(0, 72)
        splitter.setStretchFactor(1, 28)
        splitter.setChildrenCollapsible(False)
        self._central_splitter = splitter
        self.setCentralWidget(splitter)

    # ================================================================== #
    # Status bar
    # ================================================================== #
    def _build_status_bar(self) -> None:
        status = self.statusBar()

        self.coordinates_label = QLabel("X: —  Y: —  Z: —", self)
        self.coordinates_label.setMinimumWidth(220)
        status.addWidget(self.coordinates_label)

        self.zoom_label = QLabel("Zoom: 100%", self)
        self.zoom_label.setMinimumWidth(90)
        status.addPermanentWidget(self.zoom_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setVisible(False)
        status.addPermanentWidget(self.progress_bar)

        self.gpu_memory_label = QLabel("GPU: —", self)
        self.gpu_memory_label.setMinimumWidth(110)
        status.addPermanentWidget(self.gpu_memory_label)

    # ================================================================== #
    # Dock widgets
    # ================================================================== #
    def _build_docks(self) -> None:
        # DICOM browser — left.
        self.dicom_browser = DICOMBrowser("DICOM Browser", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dicom_browser)

        # Script console — bottom.
        self.script_console = ScriptConsoleWidget(parent=self)
        self.console_dock = QDockWidget("Script Console", self)
        self.console_dock.setObjectName("ScriptConsoleDock")
        self.console_dock.setWidget(self.script_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.console_dock.setVisible(False)

        # Script editor — bottom (hidden until opened from the Tools menu).
        self.script_editor = ScriptEditor(parent=self)
        self.script_editor_dock = QDockWidget("Script Editor", self)
        self.script_editor_dock.setObjectName("ScriptEditorDock")
        self.script_editor_dock.setWidget(self.script_editor)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.script_editor_dock)
        self.script_editor_dock.setVisible(False)

        # AI services — right.
        self.ai_service_panel = AIServicePanel(self)
        self.ai_dock = QDockWidget("AI Services", self)
        self.ai_dock.setObjectName("AIServiceDock")
        self.ai_dock.setWidget(self.ai_service_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)
        self.ai_dock.setVisible(False)

        self.algorithm_panel = AlgorithmPanel(self)
        self.algorithm_dock = QDockWidget("Algorithms", self)
        self.algorithm_dock.setObjectName("AlgorithmDock")
        self.algorithm_dock.setWidget(self.algorithm_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.algorithm_dock)

        # Label editor is kept in a dock so it is available without stealing
        # space from the linked viewports.
        self.label_editor = LabelEditor(self)
        self.label_editor_dock = QDockWidget("Segmentation Tools", self)
        self.label_editor_dock.setObjectName("LabelEditorDock")
        self.label_editor_dock.setWidget(self.label_editor)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.label_editor_dock)

        # Hook dock visibility to Tools-menu actions.
        self._tools_console_action.toggled.connect(self.console_dock.setVisible)
        self.console_dock.visibilityChanged.connect(
            self._tools_console_action.setChecked)
        self._tools_ai_action.toggled.connect(self.ai_dock.setVisible)
        self.ai_dock.visibilityChanged.connect(
            self._tools_ai_action.setChecked)

        # Dock toggles in the View menu.
        view_menu = None
        for action in self.menuBar().actions():
            if action.text() == "&View":
                view_menu = action.menu()
                break
        if view_menu is not None:
            view_menu.addSeparator()
            view_menu.addAction(self.dicom_browser.toggleViewAction())
            view_menu.addAction(self.console_dock.toggleViewAction())
            view_menu.addAction(self.ai_dock.toggleViewAction())
            view_menu.addAction(self.label_editor_dock.toggleViewAction())
            view_menu.addAction(self.algorithm_dock.toggleViewAction())

    # ================================================================== #
    # Controller wiring
    # ================================================================== #
    def attach_controller(self, controller) -> None:
        """Connect the UI to an ``AppController`` instance.

        All connections are guarded so that a partially implemented
        controller never crashes the UI.
        """
        self.controller = controller

        def _connect_signal(signal_name: str, slot) -> None:
            signal = getattr(controller, signal_name, None)
            if signal is not None:
                try:
                    signal.connect(slot)
                except Exception:
                    pass

        def _bind_action(action: QAction, method_name: str) -> None:
            method = getattr(controller, method_name, None)
            if callable(method):
                action.triggered.disconnect()
                action.triggered.connect(method)

        a = self.actions
        # Toolbar / menu actions -> controller slots.
        _bind_action(self.toolbar.actions_by_name["open"], "open_file")
        _bind_action(self.toolbar.actions_by_name["save"], "save_file")
        _bind_action(self.toolbar.actions_by_name["undo"], "undo")
        _bind_action(self.toolbar.actions_by_name["redo"], "redo")
        _bind_action(a["open"], "open_file")
        _bind_action(a["save"], "save_file")
        _bind_action(a["undo"], "undo")
        _bind_action(a["redo"], "redo")
        _bind_action(a["save_as"], "save_file_as")
        _bind_action(a["new"], "new_session")

        # Controller signals -> UI slots.
        _connect_signal("status_message", self.statusBar().showMessage)
        _connect_signal("progress_changed", self._on_progress)
        _connect_signal("coordinates_changed", self._on_coordinates)
        _connect_signal("zoom_changed", self._on_zoom)
        _connect_signal("gpu_memory_changed", self._on_gpu_memory)
        _connect_signal("dirty_changed", self.set_dirty)
        _connect_signal("file_opened", self._on_file_opened)
        _connect_signal("tool_changed", self.toolbar.set_tool)
        _connect_signal("label_created", self.set_active_label)

        if getattr(controller, "undo_manager", None) is not None:
            self.label_editor.set_undo_stack(controller.undo_manager)
        self.view_container.attach_label_editor(self.label_editor)
        method = getattr(controller, "on_label_modified", None)
        if callable(method):
            self.label_editor.label_modified.connect(method)

        catalog_provider = getattr(controller, "algorithm_catalog", None)
        if callable(catalog_provider):
            self.algorithm_panel.set_catalog(catalog_provider())
        schema_provider = getattr(controller, "algorithm_schema", None)
        if callable(schema_provider):
            self.algorithm_panel.set_schema_provider(schema_provider)
        run_provider = getattr(controller, "run_algorithm_request", None)
        if callable(run_provider):
            self.algorithm_panel.algorithm_run_requested.connect(run_provider)

        # Panels -> controller.
        panel = self.property_panel
        for attr, signal in (
            ("load_series", self.dicom_browser.series_load_requested),
            ("select_series", self.dicom_browser.series_selected),
            ("run_algorithm", None),
        ):
            pass  # reserved for future direct bindings

        # DICOM browser.
        method = getattr(controller, "load_series", None)
        if callable(method):
            self.dicom_browser.series_load_requested.connect(method)
        method = getattr(controller, "on_label_selected", None)
        if callable(method):
            panel.label_panel.label_selected.connect(method)
        method = getattr(controller, "on_label_visibility_changed", None)
        if callable(method):
            panel.label_panel.label_visibility_changed.connect(method)
        method = getattr(controller, "on_label_color_changed", None)
        if callable(method):
            panel.label_panel.label_color_changed.connect(method)
        method = getattr(controller, "on_export_requested", None)
        if callable(method):
            panel.export_panel.export_requested.connect(method)
        method = getattr(controller, "run_algorithm", None)
        if callable(method):
            pass  # AlgorithmPanel lives wherever the app embeds it.

        # Give the console convenient access to the controller.
        self.script_console.append_context({
            "controller": controller,
            "window": self,
        })
        try:
            from scripts.ui_integration import attach_scripts
            attach_scripts(controller, self)
        except Exception:
            pass

    # ================================================================== #
    # Comparison view
    # ================================================================== #
    def _open_comparison_view(self) -> None:
        """Open the comparison view dialog with the two most recent volumes."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        from ui.widgets.comparison_view import ComparisonView

        volumes = list(getattr(self.controller, "volumes", {}).values())
        if len(volumes) < 2:
            self.statusBar().showMessage(
                "Comparison needs at least two loaded volumes", 4000)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Comparison View")
        dialog.resize(1200, 700)
        layout = QVBoxLayout(dialog)
        view = ComparisonView(dialog)
        layout.addWidget(view)
        dialog.show()
        view.set_volumes(volumes[-2], volumes[-1])

    # ================================================================== #
    # File / session slots
    # ================================================================== #
    def set_active_label(self, label) -> None:
        """Show a label in the mask list and make it editable."""
        self.label_editor.set_label(label)
        for view in self.view_container.slice_views():
            if label.name not in getattr(view, "_overlays", {}):
                view.add_label_overlay(label)
        panel = self.property_panel.label_panel
        if panel.label_count() == 0 or panel.label_name(panel.label_count() - 1) != label.name:
            panel.add_label(name=label.name, source=label.source, visible=label.visible)

    @Slot()
    def _new_session(self) -> None:
        if not self._confirm_discard_changes():
            return
        self._current_file = None
        self.set_dirty(False)
        self.statusBar().showMessage("New session", 3000)

    @Slot()
    def _open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "Medical Images (*.dcm *.nii *.nii.gz *.nrrd *.mha);;All Files (*)")
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        controller = self.controller
        if controller is not None and hasattr(controller, "open_file"):
            controller.open_file(path)
        else:
            self.statusBar().showMessage(f"Opened {path}", 5000)
        self._add_recent_file(path)
        self._current_file = path
        self.set_dirty(False)

    @Slot()
    def _save_file(self) -> None:
        if self._current_file is None:
            self._save_file_as()
            return
        controller = self.controller
        if controller is not None and hasattr(controller, "save_file"):
            controller.save_file(self._current_file)
        self.set_dirty(False)
        self.statusBar().showMessage(f"Saved {self._current_file}", 3000)

    @Slot()
    def _save_file_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session As", "", "MedAxis Session (*.maxis);;All Files (*)")
        if not path:
            return
        self._current_file = path
        controller = self.controller
        if controller is not None and hasattr(controller, "save_file"):
            controller.save_file(path)
        self.set_dirty(False)
        self.statusBar().showMessage(f"Saved {path}", 3000)

    @Slot()
    def _export_requested(self) -> None:
        self.property_panel.set_tab("export")

    # ------------------------------------------------------------------ #
    # Recent files
    # ------------------------------------------------------------------ #
    def _add_recent_file(self, path: str) -> None:
        recents = self._settings.value(self.RECENT_FILES_KEY, [], type=list)
        if path in recents:
            recents.remove(path)
        recents.insert(0, path)
        recents = recents[: self.MAX_RECENT_FILES]
        self._settings.setValue(self.RECENT_FILES_KEY, recents)
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        self._recent_actions.clear()
        recents = self._settings.value(self.RECENT_FILES_KEY, [], type=list)
        for path in recents:
            action = QAction(path, self)
            action.triggered.connect(lambda _=False, p=path: self._load_path(p))
            self._recent_menu.addAction(action)
            self._recent_actions.append(action)
        self._recent_menu.setEnabled(bool(recents))

    # ================================================================== #
    # Edit / view slots
    # ================================================================== #
    @Slot()
    def _undo(self) -> None:
        controller = self.controller
        if controller is not None and hasattr(controller, "undo"):
            controller.undo()

    @Slot()
    def _redo(self) -> None:
        controller = self.controller
        if controller is not None and hasattr(controller, "redo"):
            controller.redo()

    @Slot()
    def _open_settings(self) -> None:
        controller = self.controller
        if controller is not None and hasattr(controller, "open_settings"):
            controller.open_settings()
        else:
            self.statusBar().showMessage("Settings dialog not implemented", 3000)

    def _set_layout(self, mode: str) -> None:
        self.view_container.set_layout_mode(mode)
        for m, action in self._layout_actions.items():
            action.setChecked(m == mode)

    def _set_tool(self, tool: str) -> None:
        if tool == "reset":
            self.view_container.reset_all_views()
            return
        self.toolbar.set_tool(tool) if tool in (
            "pointer", "paint", "erase", "fill", "distance", "angle",
            "zoom", "pan") else None
        controller = self.controller
        if controller is not None and hasattr(controller, "set_tool"):
            controller.set_tool(tool)

    # ------------------------------------------------------------------ #
    # Image menu
    # ------------------------------------------------------------------ #
    def _reset_window_level(self) -> None:
        view = self.view_container.active_slice_view()
        if view is not None:
            view.reset_window_level()

    def _apply_window_level(self, window: float, level: float) -> None:
        view = self.view_container.active_slice_view()
        if view is not None:
            view.set_window_level(window, level)

    # ------------------------------------------------------------------ #
    # Measure / Segment / Analyze slots (delegate to controller)
    # ------------------------------------------------------------------ #
    def _delegate(self, method_name: str, *args) -> None:
        controller = self.controller
        if controller is not None and hasattr(controller, method_name):
            getattr(controller, method_name)(*args)
        else:
            self.statusBar().showMessage(
                f"{method_name} not available (no controller)", 3000)

    def _roi_statistics(self) -> None:
        self._delegate("roi_statistics")

    def _segment_threshold(self) -> None:
        self._delegate("segment_threshold")

    def _segment_region_grow(self) -> None:
        self._delegate("segment_region_grow")

    def _segment_watershed(self) -> None:
        self._delegate("segment_watershed")

    def _show_histogram(self) -> None:
        self._delegate("show_histogram")

    def _show_radiomics(self) -> None:
        self._delegate("show_radiomics")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About MedAxis",
            "<h3>MedAxis — Medical Imaging Platform</h3>"
            "<p>Multi-modal medical image visualization, segmentation and "
            "analysis platform built with PySide6 and VTK.</p>"
            "<p>For research use only — not a medical device.</p>")

    # ================================================================== #
    # Status bar slots
    # ================================================================== #
    @Slot(int)
    def _on_progress(self, value: int) -> None:
        self.progress_bar.setVisible(0 < value < 100)
        self.progress_bar.setValue(max(0, min(int(value), 100)))

    def _on_coordinates(self, *args) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            self.coordinates_label.setText(args[0])
        elif len(args) >= 3:
            self.coordinates_label.setText(
                f"X: {args[0]:.1f}  Y: {args[1]:.1f}  Z: {args[2]:.1f}")

    @Slot(float)
    def _on_zoom(self, factor: float) -> None:
        self.zoom_label.setText(f"Zoom: {factor * 100:.0f}%")

    def _on_gpu_memory(self, text) -> None:
        self.gpu_memory_label.setText(f"GPU: {text}")

    def _on_file_opened(self, path: str) -> None:
        self._add_recent_file(path)
        self._current_file = path
        self.set_dirty(False)

    # ================================================================== #
    # Dirty state & close handling
    # ================================================================== #
    def set_dirty(self, dirty: bool) -> None:
        """Mark the session as having unsaved changes."""
        self._dirty = bool(dirty)
        title = "MedAxis — Medical Imaging Platform"
        if self._current_file:
            title = f"{self._current_file} — {title}"
        self.setWindowTitle(("*" if dirty else "") + title)

    def is_dirty(self) -> bool:
        return self._dirty

    def _confirm_discard_changes(self) -> bool:
        """Returns True when it is safe to proceed."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "There are unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Save:
            self._save_file()
            return not self._dirty
        return reply == QMessageBox.Discard

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_discard_changes():
            event.ignore()
            return
        # Persist window geometry.
        self._settings.setValue("geometry", self.saveGeometry())
        for view in self.view_container.slice_views():
            view.close()
        event.accept()
