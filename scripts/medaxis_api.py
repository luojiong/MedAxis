"""Stable ``medaxis`` scripting facade over the application controller."""

from __future__ import annotations

import asyncio
from typing import Any


def _run_coroutine(coro):
    """Run an async operation from a sync script, or return it in an active loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return coro


def _resolve(mapping: dict[str, Any], value: Any, current: Any = None) -> Any:
    if value is None:
        return current
    if isinstance(value, str):
        result = mapping.get(value)
        if result is None:
            raise KeyError(f"Unknown object id: {value}")
        return result
    return value


class MedAxisAPI:
    """Public object injected into user scripts as ``medaxis``."""

    def __init__(self, controller=None):
        self._controller = controller

    @property
    def data(self): return _DataAPI(self._controller)

    @property
    def io(self): return _IOAPI(self._controller)

    @property
    def views(self): return _ViewAPI(self._controller)

    @property
    def algorithms(self): return _AlgorithmAPI(self._controller)

    @property
    def ai(self): return _AIAPI(self._controller)

    @property
    def mcp(self): return _MCPAPI(self._controller)

    @property
    def ui(self): return _UIAPI(self._controller)


class _DataAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def volumes(self, id=None):
        if id is None:
            return list(getattr(self.ctrl, "volumes", {}).values())
        return _resolve(getattr(self.ctrl, "volumes", {}), id, getattr(self.ctrl, "current_volume", None))

    def labels(self, volume=None):
        manager = getattr(self.ctrl, "label_manager", None)
        labels = manager.list() if manager is not None else list(getattr(self.ctrl, "labels", {}).values())
        if volume is None:
            return labels
        volume_id = str(getattr(volume, "id", "") or id(volume))
        return [label for label in labels if label.parent_volume_id == volume_id]

    def meshes(self): return list(getattr(self.ctrl, "meshes", {}).values())


class _IOAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def _open(self, path):
        if self.ctrl.file_manager is None:
            raise RuntimeError("File manager is unavailable")
        volume = self.ctrl.file_manager.open(path)
        self.ctrl.emit_volume_loaded(volume)
        return volume

    def open_dicom(self, path): return self._open(path)
    def open_nifti(self, path): return self._open(path)

    def export_mesh(self, mesh, path):
        if self.ctrl.export_manager is None:
            raise RuntimeError("Export manager is unavailable")
        from medaxis_io.export_manager import ExportFormat
        suffix = str(path).lower().rsplit(".", 1)[-1]
        fmt = ExportFormat.OBJ if suffix == "obj" else ExportFormat.PLY if suffix == "ply" else ExportFormat.STL
        return self.ctrl.export_manager.export_mesh(mesh, fmt, path)

    def export_labelmap(self, label, path):
        if self.ctrl.export_manager is None:
            raise RuntimeError("Export manager is unavailable")
        from medaxis_io.export_manager import ExportFormat
        return self.ctrl.export_manager.export_label(label, ExportFormat.NIFTI, path, self.ctrl.current_volume)


class _ViewAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def _view(self, axis):
        container = self.ctrl.main_window.view_container
        view = container.get_view(axis)
        if view is None:
            raise RuntimeError(f"Unknown view: {axis}")
        return view

    def set_slice(self, vol, axis="axial", index=0):
        volume = _resolve(getattr(self.ctrl, "volumes", {}), vol, self.ctrl.current_volume)
        view = self._view(axis)
        view.set_volume(volume)
        view.set_slice_index(int(index))

    def set_window_level(self, vol, window=400, level=40):
        volume = _resolve(getattr(self.ctrl, "volumes", {}), vol, self.ctrl.current_volume)
        for view in self.ctrl.main_window.view_container.slice_views():
            view.set_volume(volume)
            view.set_window_level(float(window), float(level))

    def screenshot(self, path):
        view = self.ctrl.main_window.view_container.active_view()
        return self.ctrl.export_manager.export_screenshot(view, "png", path)

    def zoom_to_fit(self):
        for view in self.ctrl.main_window.view_container.slice_views():
            view.reset_view()


class _AlgorithmAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def run(self, algorithm_id, data=None, **params):
        return self.ctrl.run_algorithm(algorithm_id, data, params)

    def threshold(self, vol=None, lower=-100, upper=300):
        return self.run("manual_threshold", vol, lower=lower, upper=upper)

    def otsu(self, vol=None, num_thresholds=2):
        return self.run("otsu", vol, num_thresholds=num_thresholds)

    def region_grow(self, vol=None, seed=None, lower=0, upper=255):
        return self.run("region_growing", vol, seed=seed or [0, 0, 0], lower=lower, upper=upper)

    def flood_fill(self, vol=None, seed=None, lower=0, upper=200, connectivity=26):
        return self.run("flood_fill", vol, seed=seed or [0, 0, 0], lower=lower,
                        upper=upper, connectivity=str(connectivity))

    def watershed(self, vol=None, level=0.3): return self.run("watershed", vol, level=level)
    def gaussian(self, vol=None, sigma=1.5): return self.run("gaussian", vol, sigma=sigma)
    def morphology(self, label, operation="closing", radius=2):
        return self.run("binary_morphology", label, operation=operation, radius=radius)
    def marching_cubes(self, label, iso_value=0.5):
        return self.run("marching_cubes", label, iso_value=iso_value)

    def label_statistics(self, label, vol=None):
        return label.compute_stats(vol or self.ctrl.current_volume).to_dict()


class _AIAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def list_models(self): return _run_coroutine(self.ctrl.ai_client_manager.list_all_models())

    def segment(self, volume, model, params=None):
        return _run_coroutine(self.ctrl.ai_client_manager.segment(
            _resolve(self.ctrl.volumes, volume, self.ctrl.current_volume), model, extra_params=params or {}))

    def segment_async(self, volume, model, params=None):
        return self.ctrl.ai_client_manager.segment(
            _resolve(self.ctrl.volumes, volume, self.ctrl.current_volume), model, extra_params=params or {})

    def wait(self, job): return _run_coroutine(job) if hasattr(job, "__await__") else job


class _MCPAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def send_tool(self, name, arguments):
        if self.ctrl.mcp_server is None:
            raise RuntimeError("MCP server is unavailable")
        return _run_coroutine(self.ctrl.mcp_server.router.call_tool(name, arguments))

    def call_agent(self, prompt):
        raise NotImplementedError("Agent orchestration is provided by the external MCP client")


class _UIAPI:
    def __init__(self, ctrl): self.ctrl = ctrl

    def show_message(self, text, level="info"):
        self.ctrl.set_status(str(text))

    def progress_bar(self, value, text=""):
        self.ctrl.progress_changed.emit(int(value))
        if text:
            self.ctrl.set_status(text)

    def dialog_input(self, prompt, default=""):
        from PySide6.QtWidgets import QInputDialog
        value, accepted = QInputDialog.getText(self.ctrl.main_window, "MedAxis", prompt, text=default)
        return value if accepted else default
