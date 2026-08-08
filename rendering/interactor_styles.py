"""Custom VTK interactor styles with Qt signal bridges."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

try:
    import vtk

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "SliceInteractorStyle",
    "VolumeInteractorStyle",
    "MPRInteractorStyle",
]


class _StyleSignals(QObject):
    """Qt signal bridge shared by all interactor styles."""

    slice_index_changed = Signal(int)
    window_level_changed = Signal(float, float)
    zoom_changed = Signal(float)
    plane_changed = Signal(object, object)   # (origin, normal)
    picked_world_position = Signal(float, float, float)


def _require_vtk() -> None:
    if not HAS_VTK:
        raise RuntimeError("VTK is required for interactor styles")


if HAS_VTK:

    class SliceInteractorStyle(vtk.vtkInteractorStyleImage):
        """Slice-view interactions.

        - Mouse wheel: scroll through slices
        - Shift + wheel: zoom (camera parallel scale)
        - Ctrl + left drag: window/level adjustment
        - Left drag: pan
        """

        def __init__(self, slice_renderer=None) -> None:
            _require_vtk()
            super().__init__()
            self.signals = _StyleSignals()
            self.slice_renderer = slice_renderer  # rendering.SliceRenderer
            self._slice_index = 0
            self._max_slices = 0
            self._window = 256.0
            self._level = 128.0
            self._wl_start = None
            self.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
            self.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)
            self.AddObserver("LeftButtonPressEvent", self._on_left_press)
            self.AddObserver("MouseMoveEvent", self._on_mouse_move)
            self.AddObserver("LeftButtonReleaseEvent", self._on_left_release)

        # -- configuration --------------------------------------------
        def configure(self, slice_index: int, max_slices: int,
                      window: float, level: float) -> None:
            """Sync internal state with the slice renderer."""
            self._slice_index = slice_index
            self._max_slices = max(0, max_slices - 1)
            self._window = window
            self._level = level

        # -- wheel -----------------------------------------------------
        def _on_wheel_forward(self, _obj, _event) -> None:
            self._wheel(+1)

        def _on_wheel_backward(self, _obj, _event) -> None:
            self._wheel(-1)

        def _wheel(self, direction: int) -> None:
            interactor = self.GetInteractor()
            if interactor is None:
                return
            if interactor.GetShiftKey():
                # Zoom via camera parallel scale.
                renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
                camera = renderer.GetActiveCamera()
                scale = camera.GetParallelScale() * (0.9 if direction > 0 else 1.1)
                camera.SetParallelScale(max(scale, 1e-3))
                self.signals.zoom_changed.emit(scale)
                renderer.GetRenderWindow().Render()
                return
            self._set_slice(self._slice_index + direction)

        def _set_slice(self, index: int) -> None:
            index = max(0, min(index, self._max_slices))
            if index == self._slice_index and self._max_slices > 0:
                return
            self._slice_index = index
            if self.slice_renderer is not None:
                self.slice_renderer.update_slice(index)
            self.signals.slice_index_changed.emit(index)

        # -- window/level with ctrl-drag -------------------------------
        def _on_left_press(self, _obj, _event) -> None:
            interactor = self.GetInteractor()
            if interactor is not None and interactor.GetControlKey():
                x, y = interactor.GetEventPosition()
                self._wl_start = (x, y, self._window, self._level)
                self.StartWindowLevel()
                return
            self.OnLeftButtonDown()

        def _on_mouse_move(self, _obj, _event) -> None:
            interactor = self.GetInteractor()
            if self._wl_start is not None and interactor is not None:
                x, y = interactor.GetEventPosition()
                x0, y0, w0, l0 = self._wl_start
                # Sensitivity: full viewport drag ~ 4x current width.
                renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
                width_px = max(renderer.GetSize()[1], 1)
                self._window = max(1.0, w0 + (x - x0) * (4.0 * w0 / width_px))
                self._level = l0 + (y - y0) * (4.0 * w0 / width_px)
                if self.slice_renderer is not None:
                    self.slice_renderer.set_window_level(self._window, self._level)
                self.signals.window_level_changed.emit(self._window, self._level)
                return
            self.OnMouseMove()

        def _on_left_release(self, _obj, _event) -> None:
            if self._wl_start is not None:
                self._wl_start = None
                self.EndWindowLevel()
                return
            self.OnLeftButtonUp()

    class VolumeInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        """Standard 3D trackball interaction for the volume view."""

        def __init__(self) -> None:
            _require_vtk()
            super().__init__()
            self.signals = _StyleSignals()
            self.SetMotionFactor(10.0)
            self.SetRotationFactor(1.0)
            self.AutoAdjustCameraClippingRangeOn()

    class MPRInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        """Interactions for an oblique MPR plane.

        - Left drag: rotate the plane normal (tumbling)
        - Right drag: translate the plane along its own axes
        - Shift + left drag: in-plane rotation
        """

        def __init__(self, mpr_renderer=None) -> None:
            _require_vtk()
            super().__init__()
            self.signals = _StyleSignals()
            self.mpr_renderer = mpr_renderer  # rendering.MPRRenderer
            self._plane_drag = None
            self.AddObserver("RightButtonPressEvent", self._on_right_press)
            self.AddObserver("MouseMoveEvent", self._on_mouse_move)
            self.AddObserver("RightButtonReleaseEvent", self._on_right_release)

        def _on_right_press(self, _obj, _event) -> None:
            interactor = self.GetInteractor()
            if interactor is None or self.mpr_renderer is None:
                self.OnRightButtonDown()
                return
            x, y = interactor.GetEventPosition()
            origin, normal = self.mpr_renderer.get_current_plane()
            self._plane_drag = (x, y, origin, normal)
            self.OnRightButtonDown()

        def _on_mouse_move(self, _obj, _event) -> None:
            interactor = self.GetInteractor()
            if self._plane_drag is not None and interactor is not None and self.mpr_renderer is not None:
                x, y = interactor.GetEventPosition()
                x0, y0, origin, normal = self._plane_drag
                dy = y - y0
                # Translate: vertical drag moves along the normal.
                import numpy as np

                origin_arr = np.asarray(origin, dtype=np.float64)
                normal_arr = np.asarray(normal, dtype=np.float64)
                scale = float(self.mpr_renderer.volume.spacing_mm[0]
                              if self.mpr_renderer.volume else 1.0)
                new_origin = origin_arr + normal_arr * (-dy * scale * 0.5)
                self.mpr_renderer.set_reslice_plane(tuple(new_origin), normal)
                self._plane_drag = (x, y, tuple(new_origin), normal)
                self.signals.plane_changed.emit(tuple(new_origin), normal)
                return
            self.OnMouseMove()

        def _on_right_release(self, _obj, _event) -> None:
            self._plane_drag = None
            self.OnRightButtonUp()

        def rotate_normal(self, axis: tuple[float, float, float], angle_deg: float) -> None:
            """Rotate the plane normal programmatically (used by Shift-drag)."""
            if self.mpr_renderer is not None:
                self.mpr_renderer.rotate_plane(axis, angle_deg)
                self.signals.plane_changed.emit(*self.mpr_renderer.get_current_plane())

else:  # pragma: no cover — VTK missing

    class SliceInteractorStyle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_vtk()

    class VolumeInteractorStyle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_vtk()

    class MPRInteractorStyle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_vtk()
