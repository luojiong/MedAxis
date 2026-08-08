"""MedAxis — Central Application Controller.

Acts as the mediator between all subsystems and serves as the global
event bus using Qt signals. All cross-module communication should flow
through the AppController rather than direct imports between subsystems.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox

from utils.logging_config import setup_logging
from utils.config import AppConfig

logger = logging.getLogger(__name__)


def _volume_id(volume: Any) -> str:
    return str(getattr(volume, "id", "") or id(volume))


class AppController(QObject):
    """Central mediator / global event bus for the MedAxis application.

    Holds references to every major subsystem and exposes application-wide
    signals so that loosely-coupled components can communicate without
    importing each other.
    """

    # ------------------------------------------------------------------
    # Global event bus signals
    # ------------------------------------------------------------------
    # Emitted when a medical volume (CT/MR/PET/...) has been loaded.
    # Payload is a VolumeData object (see app.core.volume).
    volume_loaded = Signal(object)

    # Emitted when a new annotation label is created.
    label_created = Signal(object)

    # Emitted after a project file has been saved. Payload: file path.
    project_saved = Signal(str)

    # Emitted on any unhandled/recoverable error. Payload: message.
    error_occurred = Signal(str)

    # General status message for the status bar.
    status_message = Signal(str)

    # Emitted when the active project changes (open/close/switch).
    project_changed = Signal(object)

    # UI-facing state signals.  They are intentionally small and generic so
    # the controller can remain useful before every subsystem is complete.
    progress_changed = Signal(int)
    coordinates_changed = Signal(float, float, float)
    zoom_changed = Signal(float)
    gpu_memory_changed = Signal(str)
    dirty_changed = Signal(bool)
    file_opened = Signal(str)
    tool_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._initialized = False
        self.current_volume = None
        self.volumes: dict[str, Any] = {}
        self.labels: dict[str, Any] = {}
        self.meshes: dict[str, Any] = {}
        self._current_project_path: Optional[str] = None
        self._autosave_timer: Optional[QTimer] = None

        # Config & logging
        self.config: AppConfig = AppConfig.instance()

        # Subsystem references (created in initialize())
        self.workspace = None            # app.workspace.Workspace
        self.main_window = None          # app.ui.main_window.MainWindow
        self.file_manager = None         # app.core.file_manager.FileManager
        self.vtk_engine = None           # app.rendering.vtk_engine.VTKEngine
        self.algorithm_registry = None   # app.algorithms.registry.AlgorithmRegistry
        self.ai_client_manager = None    # app.ai.client_manager.AIApiclientManager
        self.mcp_server = None           # app.mcp.server.MCPServer
        self.plugin_manager = None       # app.plugins.manager.PluginManager
        self.script_engine = None        # app.scripting.engine.ScriptEngine
        self.project_manager = None      # app.core.project_manager.ProjectManager
        self.undo_manager = None         # app.core.undo_manager.UndoManager
        self.label_manager = None        # app.core.label_manager.LabelManager
        self.export_manager = None       # io.export_manager.ExportManager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create all subsystems in dependency order.

        Subsystems are imported lazily so that a missing optional module
        does not prevent the application from starting.
        """
        if self._initialized:
            logger.warning("AppController.initialize() called twice; ignoring.")
            return

        setup_logging()
        logger.info("Initializing MedAxis v0.1.0 ...")

        self.workspace = self._create_workspace()

        self._create_core()
        self._create_rendering()
        self._create_algorithms()
        self._create_ai()
        self._create_extensibility()
        self._create_ui()
        self._start_autosave()

        self._initialized = True
        logger.info("MedAxis initialization complete.")

    def show(self) -> None:
        """Show the main window."""
        if self.main_window is not None:
            self.main_window.show()
        else:
            self.report_error("Main window failed to initialize.")

    def shutdown(self) -> None:
        """Tear down subsystems in reverse order."""
        logger.info("Shutting down MedAxis ...")
        if self._autosave_timer is not None:
            self._autosave_timer.stop()
        for name in (
            "script_engine", "plugin_manager", "mcp_server",
            "ai_client_manager", "algorithm_registry", "vtk_engine",
            "file_manager", "project_manager", "main_window",
        ):
            subsystem = getattr(self, name, None)
            if subsystem is not None and hasattr(subsystem, "shutdown"):
                try:
                    subsystem.shutdown()
                except Exception:  # noqa: BLE001
                    logger.exception("Error shutting down %s", name)
        self.config.save()

    # ------------------------------------------------------------------
    # Subsystem factories (lazy imports keep startup resilient)
    # ------------------------------------------------------------------
    def _create_workspace(self):
        from app.workspace import Workspace
        ws = Workspace(self)
        ws.project_changed.connect(self._on_project_changed)
        ws.project_saved.connect(self.project_saved)
        return ws

    def _create_core(self) -> None:
        try:
            from medaxis_io.file_manager import FileManager
            self.file_manager = FileManager(self)
            self.file_manager.volume_loaded.connect(self._on_volume_loaded)
            self.file_manager.load_progress.connect(
                lambda value: self.progress_changed.emit(int(value * 100)))
            self.file_manager.load_error.connect(self.report_error)
        except (ImportError, RuntimeError) as exc:
            logger.warning("FileManager unavailable: %s", exc)

        try:
            from medaxis_io.project_io import ProjectIO
            self.project_manager = ProjectIO()
        except ImportError:
            logger.debug("Project I/O unavailable.")

        try:
            from core.undo_manager import UndoManager
            self.undo_manager = UndoManager.instance()
        except (ImportError, TypeError):
            logger.debug("UndoManager unavailable.")

        try:
            from core.label_manager import LabelManager
            self.label_manager = LabelManager()
        except ImportError:
            self.label_manager = None

        try:
            from medaxis_io.export_manager import ExportManager
            self.export_manager = ExportManager(self)
        except ImportError:
            self.export_manager = None

    def _create_rendering(self) -> None:
        try:
            from rendering.vtk_engine import VTKEngine
            self.vtk_engine = VTKEngine.instance(self)
        except (ImportError, RuntimeError) as exc:
            logger.warning("VTK unavailable: %s", exc)

    def _create_algorithms(self) -> None:
        try:
            from processing.algorithm_registry import AlgorithmRegistry
            from processing.builtins import register_builtin_algorithms
            self.algorithm_registry = AlgorithmRegistry()
            register_builtin_algorithms(self.algorithm_registry)
        except ImportError:
            logger.debug("Algorithm registry unavailable.")

    def _create_ai(self) -> None:
        try:
            from plugins.api_clients.rest_client import AIServiceManager
            self.ai_client_manager = AIServiceManager(self)
        except ImportError:
            logger.debug("AI service manager unavailable.")

    def _create_extensibility(self) -> None:
        try:
            from mcp.router import MCPRouter
            from mcp.server import MCPServer
            from mcp.tools.image_tools import register_all_tools
            from mcp.resources import register_all_resources

            router = MCPRouter()
            self.mcp_router = router
            for tool in register_all_tools(self):
                router.register_tool(tool)
            for resource in register_all_resources(self):
                router.register_resource(resource)
            self.mcp_server = MCPServer(router)
        except ImportError:
            logger.debug("MCP server unavailable.")

        try:
            from plugins.plugin_manager import PluginManager
            self.plugin_manager = PluginManager(self)
        except ImportError:
            logger.debug("Plugin manager unavailable.")

        try:
            from scripts.medaxis_api import MedAxisAPI
            self.script_engine = MedAxisAPI(self)
        except ImportError:
            self.script_engine = None

    def _create_ui(self) -> None:
        try:
            from ui.main_window import MainWindow
            self.main_window = MainWindow(self)
        except (ImportError, RuntimeError) as exc:
            logger.exception("Main window failed to initialize: %s", exc)

    # ------------------------------------------------------------------
    # Event bus helpers
    # ------------------------------------------------------------------
    @Slot(object)
    def _on_project_changed(self, project: Any) -> None:
        self.project_changed.emit(project)

    def emit_volume_loaded(self, volume: Any) -> None:
        """Convenience entry point for subsystems to announce a volume."""
        logger.info("Volume loaded: %r", volume)
        self._on_volume_loaded(volume)

    @Slot(object)
    def _on_volume_loaded(self, volume: Any) -> None:
        """Distribute a loaded volume to the view layer and the event bus."""
        self.current_volume = volume
        volume_id = str(getattr(volume, "id", "") or id(volume))
        self.volumes[volume_id] = volume
        self.volume_loaded.emit(volume)
        if self.main_window is not None:
            self.main_window.view_container.set_volume(volume)
            self.main_window.set_dirty(True)
        self.progress_changed.emit(100)
        self.status_message.emit(
            f"Loaded {getattr(volume, 'name', 'volume')} "
            f"{getattr(volume, 'dimensions', '')}")

    def emit_label_created(self, label: Any) -> None:
        self.label_created.emit(label)

    @Slot(str)
    def report_error(self, message: str) -> None:
        """Global error reporting slot — any subsystem may call this."""
        logger.error(message)
        self.error_occurred.emit(message)

    @Slot(str)
    def set_status(self, message: str) -> None:
        self.status_message.emit(message)

    # ------------------------------------------------------------------
    # Main-window command surface
    # ------------------------------------------------------------------
    def open_file(self, path=None) -> None:
        """Open a medical image, showing the file dialog when called by Qt."""
        if isinstance(path, bool) or path is None:
            if self.main_window is None:
                return
            path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Open Medical Image",
                "",
                "Medical Images (*.dcm *.nii *.nii.gz *.nrrd *.nhdr *.raw *.medaxis);;"
                "All Files (*)",
            )
        if not path:
            return
        if str(path).lower().endswith(".medaxis"):
            self.open_project(str(path))
            return
        if self.file_manager is None:
            return
        self.status_message.emit(f"Loading {path} ...")
        self.progress_changed.emit(1)
        self.file_manager.open_queued(str(path))
        self.file_opened.emit(str(path))

    def save_file(self, path=None) -> bool:
        """Persist the current session metadata when a project path exists."""
        if isinstance(path, bool) or path is None:
            path = getattr(self, "_current_project_path", None)
        if not path:
            return self.save_file_as()
        if self.current_volume is None or self.project_manager is None:
            self.status_message.emit("Nothing to save")
            return False
        try:
            from core.project import Project
            project = Project()
            project.path = str(path)
            project.views = self.main_window.view_container.to_state() if self.main_window else {}
            project.labels = self.label_manager.metadata() if self.label_manager is not None else []
            project.meshes = [mesh.to_dict() for mesh in self.meshes.values()
                              if hasattr(mesh, "to_dict")]
            self.project_manager.save_project(
                project,
                path,
                self.volumes,
                labels=self.labels,
                meshes=self.meshes,
            )
            self._current_project_path = str(path)
            self.dirty_changed.emit(False)
            self.project_saved.emit(str(path))
            return True
        except Exception as exc:  # noqa: BLE001
            self.report_error(f"Failed to save session: {exc}")
            return False

    def _start_autosave(self) -> None:
        interval_sec = int(self.config.get("ui", "auto_save_interval_seconds", 0) or 0)
        if interval_sec <= 0 or self.project_manager is None:
            return
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(interval_sec * 1000)
        self._autosave_timer.timeout.connect(self._autosave_current_session)
        self._autosave_timer.start()

    def _autosave_current_session(self) -> None:
        if self.current_volume is None or self.project_manager is None:
            return
        try:
            from core.project import Project
            project = Project(labels=self.label_manager.metadata() if self.label_manager else [])
            self.project_manager.autosave(
                project,
                volumes=self.volumes,
                labels=self.labels,
                meshes=self.meshes,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Autosave failed")

    def save_file_as(self, *_args) -> bool:
        if self.main_window is None:
            return False
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save MedAxis Session", "", "MedAxis (*.medaxis)")
        return self.save_file(path) if path else False

    def new_session(self, *_args) -> None:
        self.current_volume = None
        self.volumes.clear()
        self.labels.clear()
        self.meshes.clear()
        if self.label_manager is not None:
            self.label_manager.clear()
        self._current_project_path = None
        if self.main_window is not None:
            self.main_window.view_container.reset_all_views()
            self.main_window.property_panel.label_panel.clear()
            self.main_window.property_panel.clear_meshes()
            self.main_window.set_dirty(False)
        self.status_message.emit("New session")

    def undo(self) -> None:
        if self.undo_manager is not None:
            self.undo_manager.undo()

    def redo(self) -> None:
        if self.undo_manager is not None:
            self.undo_manager.redo()

    def reset_views(self) -> None:
        if self.main_window is not None:
            self.main_window.view_container.reset_all_views()

    def set_tool(self, tool: str) -> None:
        self.tool_changed.emit(str(tool))

    def open_settings(self) -> None:
        QMessageBox.information(self.main_window, "Settings", "Settings are stored in ~/.medaxis/config.yaml.")

    def load_series(self, path: str) -> None:
        self.open_file(path)

    def run_algorithm(self, algorithm_id: str, input_data=None, params: Optional[dict] = None):
        """Run a registered algorithm and attach any generated result to the workspace."""
        if self.algorithm_registry is None:
            raise RuntimeError("Algorithm registry is unavailable")
        data = input_data if input_data is not None else self.current_volume
        result = self.algorithm_registry.run(algorithm_id, data, params or {})
        if not result.success:
            self.report_error(result.error or f"Algorithm failed: {algorithm_id}")
            return result

        if result.label_data is not None and self.label_manager is not None:
            parent_id = _volume_id(self.current_volume) if self.current_volume is not None else ""
            label = self.label_manager.create(
                np.asarray(result.label_data),
                parent_id,
                name=algorithm_id,
                source=f"algorithm:{algorithm_id}",
            )
            self.labels[label.id] = label
            result.label_data = label
            self.label_created.emit(label)
        if result.volume_data is not None:
            from core.volume_data import VolumeData

            output = result.volume_data
            if not isinstance(output, VolumeData):
                output = VolumeData(
                    array=np.asarray(output),
                    spacing=self.current_volume.spacing_mm,
                    origin=self.current_volume.origin_mm,
                    direction=self.current_volume.direction,
                    name=f"{self.current_volume.name}_{algorithm_id}",
                    modality=self.current_volume.modality,
                )
            self.current_volume = output
            self.volumes[_volume_id(output)] = output
            result.volume_data = output
            if self.main_window is not None:
                self.main_window.view_container.set_volume(output)
        if result.mesh_data is not None:
            mesh_id = str(getattr(result.mesh_data, "id", "") or id(result.mesh_data))
            self.meshes[mesh_id] = result.mesh_data
        self.dirty_changed.emit(True)
        return result

    def algorithm_catalog(self) -> dict[str, list[str]]:
        if self.algorithm_registry is None:
            return {}
        catalog: dict[str, list[str]] = {}
        for definition in self.algorithm_registry.list_all():
            catalog.setdefault(definition.category.value, []).append(definition.id)
        return catalog

    def algorithm_schema(self, _category: str, algorithm_id: str) -> list[dict]:
        definition = self.algorithm_registry.get(algorithm_id) if self.algorithm_registry else None
        if definition is None:
            return []
        return [
            {
                "name": parameter.name,
                "label": parameter.label,
                "type": parameter.type,
                "default": parameter.default,
                "min": parameter.min_val,
                "max": parameter.max_val,
                "options": parameter.choices or [],
            }
            for parameter in definition.parameters
        ]

    def run_algorithm_request(self, payload: dict) -> None:
        """Adapter for AlgorithmPanel's UI payload."""
        algorithm_id = payload.get("algorithm")
        definition = self.algorithm_registry.get(algorithm_id) if self.algorithm_registry else None
        input_data = self.current_volume
        if definition is not None and definition.input_parameter == "label_data":
            labels = self.label_manager.for_volume(_volume_id(self.current_volume)) if self.label_manager else []
            input_data = labels[-1] if labels else None
        elif definition is not None and definition.input_parameter == "mesh_data":
            input_data = list(self.meshes.values())[-1] if self.meshes else None
        if input_data is None:
            self.report_error(f"No input data available for {algorithm_id}")
            return
        result = self.run_algorithm(algorithm_id, input_data=input_data,
                                    params=payload.get("params", {}))
        if result.success:
            self.status_message.emit(f"Completed {algorithm_id} in {result.execution_time_sec:.2f}s")

    def open_project(self, path: str) -> bool:
        if self.project_manager is None:
            return False
        try:
            project = self.project_manager.load_project(path)
            self._current_project_path = str(path)
            self.workspace._set_active(project)
            restored_volumes = self.project_manager.load_volumes(project)
            self.volumes.clear()
            for volume in restored_volumes.values():
                self._on_volume_loaded(volume)
            self._restore_project_data(project)
            if self.main_window is not None:
                self.main_window.view_container.restore_state(getattr(project, "views", {}))
            return True
        except Exception as exc:  # noqa: BLE001
            self.report_error(f"Failed to open project: {exc}")
            return False

    def _restore_project_data(self, project) -> None:
        self.labels.clear()
        if self.label_manager is not None:
            self.label_manager.clear()
        base_dir = getattr(project, "base_dir", None)
        for entry in getattr(project, "labels", []):
            file_ref = entry.get("file")
            if not file_ref or base_dir is None:
                continue
            try:
                from pathlib import Path
                source = Path(base_dir) / file_ref
                with np.load(source) as data:
                    array = data["array"]
                label = self.label_manager.create(
                    array, entry.get("parent_volume_id", ""), entry.get("name", "Label"),
                    entry.get("source", "project"), label_id=entry.get("id"),
                )
                self.labels[label.id] = label
                self.label_created.emit(label)
            except (OSError, KeyError, ValueError):
                logger.warning("Unable to restore label %s", entry.get("id"))

    def on_label_selected(self, row: int) -> None:
        if self.label_manager is None:
            return
        labels = self.label_manager.list()
        if 0 <= row < len(labels) and self.main_window is not None:
            self.main_window.set_active_label(labels[row])

    def on_label_visibility_changed(self, row: int, visible: bool) -> None:
        if self.label_manager is not None:
            labels = self.label_manager.list()
            if 0 <= row < len(labels) and self.main_window is not None:
                label = labels[row]
                label.visible = bool(visible)
                for view in self.main_window.view_container.slice_views():
                    view.set_label_visibility(label, visible)
        self.dirty_changed.emit(True)

    def on_label_boolean(self, row: int, operation: str) -> None:
        """Union / intersect / subtract between the first two labels."""
        import numpy as np

        labels = list(self.label_manager.list())
        if len(labels) < 2:
            self.report_error("Boolean operations need at least two labels")
            return
        a, b = labels[0], labels[1]
        arr_a = np.asarray(a.array) > 0
        arr_b = np.asarray(b.array) > 0
        if arr_a.shape != arr_b.shape:
            self.report_error("Labels must have the same shape")
            return
        if operation == "union":
            result = arr_a | arr_b
        elif operation == "intersect":
            result = arr_a & arr_b
        elif operation == "subtract":
            result = arr_a & ~arr_b
        else:
            self.report_error(f"Unknown boolean operation: {operation}")
            return
        out = self.label_manager.create(
            result.astype(np.int16), a.parent_volume_id,
            name=f"{a.name}_{operation}_{b.name}", source="label:boolean")
        self.labels[out.id] = out
        self.label_created.emit(out)
        self.set_status(f"Label {operation} complete: {out.name}")

    def on_label_color_changed(self, row: int, color) -> None:
        if self.label_manager is not None:
            labels = self.label_manager.list()
            if 0 <= row < len(labels):
                labels[row].color = (color.red(), color.green(), color.blue(), color.alpha())
        self.dirty_changed.emit(True)

    def on_label_modified(self, label: Any) -> None:
        if self.main_window is not None:
            for view in self.main_window.view_container.slice_views():
                view.refresh_overlay(label)
        self.dirty_changed.emit(True)

    def on_export_requested(self, _format: str) -> None:
        if self.main_window is None or self.export_manager is None:
            self.report_error("Export manager is unavailable")
            return
        format_key = str(_format).lower()
        extension = {
            "nifti": ".nii.gz", "nrrd": ".nrrd", "dicom_seg": ".dcm",
            "stl": ".stl", "obj": ".obj", "ply": ".ply", "png": ".png", "pdf": ".pdf",
        }.get(format_key, "")
        path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Export MedAxis Data",
            f"{getattr(self.current_volume, 'name', 'medaxis')}{extension}",
            f"*{extension}" if extension else "All Files (*)",
        )
        if not path:
            return
        try:
            if format_key in ("nifti", "nrrd"):
                output = self.export_manager.export_volume(self.current_volume, format_key, path)
            elif format_key == "dicom_seg":
                labels = self.label_manager.for_volume(_volume_id(self.current_volume)) if self.label_manager else []
                output = self.export_manager.export_label(labels[-1], format_key, path, self.current_volume) if labels else None
            elif format_key in ("stl", "obj", "ply"):
                mesh = list(self.meshes.values())[-1] if self.meshes else None
                output = self.export_manager.export_mesh(mesh, format_key, path) if mesh else None
            elif format_key == "png":
                view = self.main_window.view_container.active_view()
                output = self.export_manager.export_screenshot(view, format_key, path)
            elif format_key == "pdf":
                from core.project import Project
                project = Project(labels=self.label_manager.metadata() if self.label_manager else [])
                output = self.export_manager.export_report(project, format_key, path)
            else:
                raise ValueError(f"Unsupported export format: {format_key}")
            if output:
                self.status_message.emit(f"Exported {output}")
        except Exception as exc:  # noqa: BLE001
            self.report_error(f"Export failed: {exc}")

    def _delegate_status(self, label: str) -> None:
        self.status_message.emit(f"{label} is not available for the current volume")

    def roi_statistics(self):
        labels = self.label_manager.for_volume(_volume_id(self.current_volume)) if self.label_manager and self.current_volume else []
        if labels:
            stats = labels[-1].compute_stats(self.current_volume).to_dict()
            self.status_message.emit(f"{labels[-1].name}: {stats['volume_ml']:.3f} ml")
            return stats
        if self.current_volume is None:
            return self._delegate_status("ROI statistics")
        lo, hi = self.current_volume.get_min_max()
        self.status_message.emit(f"Volume range: {lo:.2f} .. {hi:.2f}")
        return {"min": lo, "max": hi}

    def segment_threshold(self): return self.run_algorithm("manual_threshold", params={"lower": -100, "upper": 300})
    def segment_region_grow(self): return self.run_algorithm("region_growing", params={"seed": [0, 0, 0], "lower": 0, "upper": 255})
    def segment_watershed(self): return self.run_algorithm("watershed", params={})

    def show_histogram(self):
        if self.current_volume is None:
            return self._delegate_status("Histogram")
        counts, _edges = self.current_volume.histogram()
        self.status_message.emit(f"Histogram ready: {len(counts)} bins")
        return counts

    def show_radiomics(self):
        result = self.run_algorithm("first_order_radiomics", params={})
        if result.success:
            self.status_message.emit("Radiomics features calculated")
        return result
