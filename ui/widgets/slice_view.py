"""2D medical slice viewer for MedAxis.

:class:`SliceView` embeds a :class:`QVTKRenderWindowInteractor` and renders a
single orthogonal slice (axial / coronal / sagittal) of a volumetric dataset
through a ``vtkImageReslice`` pipeline.

Interaction model
-----------------
======================  =======================================
Input                   Action
======================  =======================================
Scroll wheel            Navigate slices
Ctrl + Scroll wheel     Zoom
Middle button drag      Pan
Left button drag        Window / level
Left click (pointer)    Move linked crosshair
C                       Toggle cine playback
======================  =======================================

The view also supports label overlays (colour-mapped mask images), polydata
overlays (e.g. CPR reference lines), direction markers, a scale bar, a slice
position indicator and a linked crosshair.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray, vtkImageData, vtkPolyData,
)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkImageActor, vtkPolyDataMapper, vtkRenderer, vtkTextActor,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support

# Rendering back-ends (side-effect imports).
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401

Vec3 = Tuple[float, float, float]

#: Reslice axes matrices for the three canonical orientations
#: (rows: X-axis, Y-axis, Z-axis/normal, origin).
RESLICE_MATRICES = {
    "axial": (1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1),
    "coronal": (1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1),
    "sagittal": (0, 0, -1, 0,
                 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 0, 1),
}

#: World-space axis index of the slice normal per orientation.
NORMAL_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

#: Anatomical direction markers shown at the viewport edges
#: (radiological convention: patient's right appears on the image left).
DIRECTION_MARKERS = {
    "axial": {"top": "A", "bottom": "P", "left": "R", "right": "L"},
    "coronal": {"top": "S", "bottom": "I", "left": "R", "right": "L"},
    "sagittal": {"top": "S", "bottom": "I", "left": "A", "right": "P"},
}


class SliceView(QWidget):
    """Single orthogonal slice viewer with VTK rendering.

    Parameters
    ----------
    orientation:
        One of ``"axial"``, ``"coronal"``, ``"sagittal"``.

    Signals
    -------
    slice_changed(int, str)
        Current slice index and orientation.
    crosshair_moved(tuple)
        World-space crosshair position changed by user interaction.
    zoom_changed(float)
        Current zoom factor (1.0 == fit-to-view).
    left_clicked(tuple)
        World-space position of a plain left click (pointer tool).
    """

    slice_changed = Signal(int, str)
    crosshair_moved = Signal(tuple)
    zoom_changed = Signal(float)
    left_clicked = Signal(tuple)

    def __init__(self, orientation: str = "axial",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        if orientation not in RESLICE_MATRICES:
            raise ValueError(f"Unknown orientation: {orientation}")
        self.setObjectName(f"SliceView_{orientation}")
        self.setFocusPolicy(Qt.StrongFocus)

        # ------------------------------------------------------------- #
        # State
        # ------------------------------------------------------------- #
        self._orientation = orientation
        self._image: Optional[vtkImageData] = None       # source volume
        self._volume_obj = None                          # python wrapper object
        self._np_volume: Optional[np.ndarray] = None     # cached numpy view
        self._slice_index = 0
        self._num_slices = 1
        self._window: Optional[float] = None
        self._level: Optional[float] = None
        self._crosshair: Vec3 = (0.0, 0.0, 0.0)
        self._zoom_factor = 1.0
        self._cine_timer = QTimer(self)
        self._cine_timer.timeout.connect(self._cine_tick)
        self._label_editor = None                        # optional LabelEditor
        self._overlays: dict = {}                        # name -> overlay dict
        self._poly_overlays: List[vtkActor] = []
        self._brush_actor: Optional[vtkActor] = None

        # Interaction state.
        self._left_pressed = False
        self._middle_pressed = False
        self._last_pos: Tuple[int, int] = (0, 0)
        self._wl_start: Tuple[float, float] = (400.0, 40.0)
        self._press_pos: Tuple[int, int] = (0, 0)

        # ------------------------------------------------------------- #
        # VTK pipeline
        # ------------------------------------------------------------- #
        self._axes_matrix = vtkMatrix4x4()
        self._axes_matrix.DeepCopy(RESLICE_MATRICES[orientation])

        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.07, 0.07, 0.09)

        self._iren = QVTKRenderWindowInteractor(self)
        self._iren.Initialize()
        self._iren.SetInteractorStyle(vtkInteractorStyleUser())
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        self._reslice = vtkImageReslice()
        self._reslice.SetResliceAxes(self._axes_matrix)
        self._reslice.SetInterpolationModeToLinear()
        self._reslice.SetOutputDimensionality(2)

        self._image_actor = vtkImageActor()
        self._image_actor.GetMapper().SetInputConnection(
            self._reslice.GetOutputPort())
        self._image_actor.InterpolateOff()
        self._renderer.AddActor(self._image_actor)

        self._camera = self._renderer.GetActiveCamera()
        self._camera.ParallelProjectionOn()

        self._build_overlays()
        self._install_observers()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._iren)

    # ================================================================== #
    # Construction helpers
    # ================================================================== #
    def _build_overlays(self) -> None:
        """Create crosshair, direction markers, scale bar and info text."""
        # Crosshair: two lines crossing at the cursor position.
        pts = vtkPoints()
        pts.SetNumberOfPoints(4)
        lines = vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        pd = vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        self._crosshair_points = pts
        self._crosshair_actor = vtkActor()
        self._crosshair_actor.SetMapper(vtkPolyDataMapper())
        self._crosshair_actor.GetMapper().SetInputData(pd)
        self._crosshair_actor.GetProperty().SetColor(0.2, 0.9, 0.3)
        self._crosshair_actor.GetProperty().SetLineWidth(1.0)
        self._crosshair_actor.SetVisibility(False)
        self._renderer.AddActor(self._crosshair_actor)

        # Direction markers.
        self._dir_actors = {}
        positions = {
            "top": (0.49, 0.93), "bottom": (0.49, 0.02),
            "left": (0.02, 0.49), "right": (0.95, 0.49),
        }
        for key, (x, y) in positions.items():
            actor = vtkTextActor()
            actor.SetInput("")
            actor.SetDisplayPosition(int(x * 600), int(y * 600))
            actor.GetTextProperty().SetFontSize(16)
            actor.GetTextProperty().SetColor(1.0, 0.75, 0.2)
            actor.GetTextProperty().SetBold(True)
            self._renderer.AddActor2D(actor)
            self._dir_actors[key] = actor

        # Slice position / zoom indicator (bottom-right).
        self._info_actor = vtkTextActor()
        self._info_actor.SetDisplayPosition(8, 8)
        self._info_actor.GetTextProperty().SetFontSize(13)
        self._info_actor.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        self._renderer.AddActor2D(self._info_actor)

        # Scale bar (top-left text; the line itself is a small actor).
        self._scale_text = vtkTextActor()
        self._scale_text.SetDisplayPosition(8, 30)
        self._scale_text.GetTextProperty().SetFontSize(12)
        self._scale_text.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        self._renderer.AddActor2D(self._scale_text)

        self._update_direction_markers()

    def _install_observers(self) -> None:
        iren = self._iren
        iren.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        iren.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)
        iren.AddObserver("LeftButtonPressEvent", self._on_left_press)
        iren.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        iren.AddObserver("MouseMoveEvent", self._on_mouse_move)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        iren.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        iren.AddObserver("KeyPressEvent", self._on_key_press)

    # ================================================================== #
    # Public API
    # ================================================================== #
    def set_volume(self, volume) -> None:
        """Set the volume to display.

        *volume* may be a ``vtkImageData`` directly or any object exposing
        an ``image_data`` attribute (e.g. the application's ``VolumeData``).
        """
        if volume is None:
            return
        image = volume
        if not isinstance(volume, vtkImageData):
            image = getattr(volume, "image_data", None)
            if image is None:
                raise TypeError("volume must be vtkImageData or expose "
                                "'image_data'")
        self._volume_obj = volume
        self._image = image
        self._np_volume = None

        self._reslice.SetInputData(image)
        self._reslice.Update()

        extent = image.GetExtent()
        axis = NORMAL_AXIS[self._orientation]
        self._num_slices = extent[2 * axis + 1] - extent[2 * axis] + 1

        # Initial window/level from the scalar range.
        lo, hi = image.GetScalarRange()
        self._window = max(hi - lo, 1.0)
        self._level = (hi + lo) / 2.0
        self._image_actor.SetWindow(self._window)
        self._image_actor.SetLevel(self._level)

        # Start in the middle of the stack.
        self.set_slice_index(self._num_slices // 2, emit=False)

        # Overlay pipelines need the new input.
        for overlay in self._overlays.values():
            overlay["reslice"].Update()

        self.reset_view()
        self.render()

    def set_orientation(self, orientation: str) -> None:
        """Switch between axial / coronal / sagittal."""
        if orientation not in RESLICE_MATRICES:
            raise ValueError(f"Unknown orientation: {orientation}")
        self._orientation = orientation
        self._axes_matrix.DeepCopy(RESLICE_MATRICES[orientation])
        if self._image is not None:
            extent = self._image.GetExtent()
            axis = NORMAL_AXIS[orientation]
            self._num_slices = extent[2 * axis + 1] - extent[2 * axis] + 1
            self.set_slice_index(self._num_slices // 2, emit=False)
        self._update_direction_markers()
        self.render()

    def orientation(self) -> str:
        return self._orientation

    def normal_axis(self) -> int:
        """World-space axis index (0=x, 1=y, 2=z) of the slice normal."""
        return NORMAL_AXIS[self._orientation]

    def set_slice_index(self, index: int, emit: bool = True) -> None:
        """Move to slice *index* along the view normal."""
        if self._image is None:
            return
        index = max(0, min(int(index), self._num_slices - 1))
        self._slice_index = index

        axis = self.normal_axis()
        origin = self._image.GetOrigin()
        spacing = self._image.GetSpacing()
        extent = self._image.GetExtent()
        center_index = (extent[2 * axis] + extent[2 * axis + 1]) / 2.0

        plane_pos = [origin[i] + center_index * spacing[i] for i in range(3)]
        plane_pos[axis] = origin[axis] + \
            (extent[2 * axis] + index) * spacing[axis]
        for i in range(3):
            self._axes_matrix.SetElement(i, 3, plane_pos[i])

        self._reslice.Update()
        for overlay in self._overlays.values():
            overlay["reslice"].Update()

        # Keep crosshair on the new plane.
        ch = list(self._crosshair)
        ch[axis] = plane_pos[axis]
        self._crosshair = tuple(ch)
        self._update_crosshair_actor()
        self._update_info_text()
        self.render()
        if emit:
            self.slice_changed.emit(index, self._orientation)

    def slice_index(self) -> int:
        return self._slice_index

    def num_slices(self) -> int:
        return self._num_slices

    def set_window_level(self, window: float, level: float) -> None:
        """Set the display window (width) and level."""
        self._window = float(window)
        self._level = float(level)
        self._image_actor.SetWindow(self._window)
        self._image_actor.SetLevel(self._level)
        self.render()

    def get_window_level(self) -> Tuple[float, float]:
        return (self._window or 1.0, self._level or 0.0)

    def reset_window_level(self) -> None:
        if self._image is not None:
            lo, hi = self._image.GetScalarRange()
            self.set_window_level(max(hi - lo, 1.0), (hi + lo) / 2.0)

    def reset_view(self) -> None:
        """Fit the current slice to the viewport."""
        self._renderer.ResetCamera()
        self._camera.ParallelProjectionOn()
        out = self._reslice.GetOutput()
        if out is not None and out.GetNumberOfPoints():
            ext = out.GetExtent()
            sp = out.GetSpacing()
            height = (ext[3] - ext[2] + 1) * sp[1]
            if height > 0:
                self._fit_parallel_scale = height / 2.0
                self._camera.SetParallelScale(self._fit_parallel_scale * 1.05)
        self._zoom_factor = 1.0
        self._update_info_text()
        self.render()

    # ------------------------------------------------------------------ #
    # Crosshair / linked views
    # ------------------------------------------------------------------ #
    def set_cursor(self, world: Sequence[float], emit: bool = False) -> None:
        """Place the crosshair at *world* without moving the slice stack."""
        self._crosshair = (float(world[0]), float(world[1]),
                           float(world[2]))
        self._crosshair_actor.SetVisibility(True)
        self._update_crosshair_actor()
        self.render()
        if emit:
            self.crosshair_moved.emit(self._crosshair)

    def get_cursor(self) -> Vec3:
        return self._crosshair

    def get_slice_world_position(self) -> float:
        """World-space coordinate of the current plane along its normal."""
        return self._crosshair[self.normal_axis()]

    def _on_linked_crosshair(self, world: Sequence[float]) -> None:
        self.set_cursor(world, emit=False)

    # ------------------------------------------------------------------ #
    # Overlays
    # ------------------------------------------------------------------ #
    def add_label_overlay(self, label) -> None:
        """Overlay a ``LabelData``-like object (``.data`` numpy 3-D array,
        ``.name``, ``.color``, optional ``.value``) on this view."""
        name = getattr(label, "name", str(id(label)))
        data = np.asarray(label.data)
        if self._image is None or data.shape != tuple(
                self._image.GetDimensions()[::-1]):
            # Fall back to volume dimensions if available.
            if self._image is not None:
                pass

        image = vtkImageData()
        image.CopyStructure(self._image) if self._image else None
        if self._image is not None:
            image.SetDimensions(self._image.GetDimensions())
            image.SetSpacing(self._image.GetSpacing())
            image.SetOrigin(self._image.GetOrigin())
        arr = numpy_support.numpy_to_vtk(
            data.astype(np.uint8).ravel(), deep=True)
        image.GetPointData().SetScalars(arr)

        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(0, 255)
        color = getattr(label, "color", QColor(255, 0, 0))
        rgb = (color.redF(), color.greenF(), color.blueF())
        for i in range(256):
            if i == 0:
                lut.SetTableValue(i, 0.0, 0.0, 0.0, 0.0)
            else:
                lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
        lut.Build()

        reslice = vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxes(self._axes_matrix)   # shared matrix!
        reslice.SetInterpolationModeToNearestNeighbor()
        reslice.SetOutputDimensionality(2)

        actor = vtkImageActor()
        actor.GetMapper().SetInputConnection(reslice.GetOutputPort())
        actor.SetOpacity(0.55)
        actor.GetProperty().SetLookupTable(lut)
        actor.GetProperty().UseLookupTableScalarRangeOn()
        self._renderer.AddActor(actor)

        self._overlays[name] = {
            "label": label, "image": image, "reslice": reslice,
            "actor": actor, "lut": lut,
        }
        self.render()

    def refresh_overlay(self, label_or_name) -> None:
        """Redraw an overlay after its voxel data changed."""
        name = getattr(label_or_name, "name", label_or_name)
        overlay = self._overlays.get(name)
        if overlay is None:
            return
        data = np.asarray(overlay["label"].data)
        arr = numpy_support.numpy_to_vtk(
            data.astype(np.uint8).ravel(), deep=True)
        overlay["image"].GetPointData().SetScalars(arr)
        overlay["image"].Modified()
        overlay["reslice"].Update()
        self.render()

    def set_label_visibility(self, label_or_name, visible: bool) -> None:
        name = getattr(label_or_name, "name", label_or_name)
        overlay = self._overlays.get(name)
        if overlay is not None:
            overlay["actor"].SetVisibility(bool(visible))
            self.render()

    def remove_label_overlay(self, label_or_name) -> None:
        name = getattr(label_or_name, "name", label_or_name)
        overlay = self._overlays.pop(name, None)
        if overlay is not None:
            self._renderer.RemoveActor(overlay["actor"])
            self.render()

    def add_polydata_overlay(self, polydata: vtkPolyData,
                             color=(1.0, 1.0, 0.0)) -> vtkActor:
        """Draw a polydata (line/curve) overlay in world coordinates."""
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(2.0)
        self._renderer.AddActor(actor)
        self._poly_overlays.append(actor)
        self.render()
        return actor

    def clear_polydata_overlays(self) -> None:
        for actor in self._poly_overlays:
            self._renderer.RemoveActor(actor)
        self._poly_overlays.clear()
        self.render()

    # ------------------------------------------------------------------ #
    # Label editor integration
    # ------------------------------------------------------------------ #
    def attach_label_editor(self, editor) -> None:
        """Attach a :class:`LabelEditor`; its tools intercept left-drag."""
        self._label_editor = editor
        editor.set_host(self)

    def show_brush_cursor(self, world: Sequence[float],
                          radius_mm: float) -> None:
        """Show a circular brush cursor centred at *world*."""
        from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
        src = vtkRegularPolygonSource()
        src.SetNumberOfSides(32)
        src.SetRadius(max(radius_mm, 0.1))
        src.GeneratePolygonOff()
        normal = [0.0, 0.0, 0.0]
        normal[self.normal_axis()] = 1.0
        src.SetNormal(*normal)
        src.SetCenter(*world)
        src.Update()
        if self._brush_actor is None:
            self._brush_actor = vtkActor()
            self._brush_actor.GetProperty().SetColor(0.3, 0.8, 1.0)
            self._brush_actor.GetProperty().SetLineWidth(1.5)
            self._renderer.AddActor(self._brush_actor)
        self._brush_actor.GetMapper().SetInputData(src.GetOutput())
        self._brush_actor.SetVisibility(True)
        self.render()

    def hide_brush_cursor(self) -> None:
        if self._brush_actor is not None:
            self._brush_actor.SetVisibility(False)
            self.render()

    # ------------------------------------------------------------------ #
    # Cine mode
    # ------------------------------------------------------------------ #
    def start_cine(self, fps: float = 10.0) -> None:
        """Auto-play through slices at *fps* frames per second."""
        self._cine_timer.start(max(int(1000 / max(fps, 0.1)), 1))

    def stop_cine(self) -> None:
        self._cine_timer.stop()

    def cine_running(self) -> bool:
        return self._cine_timer.isActive()

    def _cine_tick(self) -> None:
        next_index = (self._slice_index + 1) % self._num_slices
        self.set_slice_index(next_index)

    # ------------------------------------------------------------------ #
    # Data access helpers (used by LabelEditor et al.)
    # ------------------------------------------------------------------ #
    def get_volume_array(self) -> Optional[np.ndarray]:
        """Return the volume as a numpy array shaped (z, y, x)."""
        if self._image is None:
            return None
        if self._np_volume is None:
            scalars = self._image.GetPointData().GetScalars()
            arr = numpy_support.vtk_to_numpy(scalars)
            dims = self._image.GetDimensions()  # (x, y, z)
            self._np_volume = arr.reshape(dims[2], dims[1], dims[0])
        return self._np_volume

    def get_volume_spacing(self) -> Vec3:
        if self._image is not None:
            return tuple(self._image.GetSpacing())  # type: ignore[return-value]
        return (1.0, 1.0, 1.0)

    def render(self) -> None:
        self._iren.GetRenderWindow().Render()

    # ================================================================== #
    # Coordinate conversion
    # ================================================================== #
    def _display_to_world(self, x: int, y: int) -> Vec3:
        """Convert VTK display coordinates to world coordinates."""
        from vtkmodules.vtkCommonCore import vtkCoordinate
        coord = vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetValue(x, y, 0)
        world = coord.GetComputedWorldValue(self._renderer)
        return (world[0], world[1], world[2])

    def _world_to_slice_uv(self, world: Sequence[float]) -> Tuple[float, float]:
        """Project a world point onto the slice plane (data coordinates)."""
        inv = vtkMatrix4x4()
        vtkMatrix4x4.Invert(self._axes_matrix, inv)
        p = inv.MultiplyPoint((world[0], world[1], world[2], 1.0))
        return p[0], p[1]

    def _display_to_voxel(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """Map display coords to voxel indices (i, j, k) or None."""
        if self._image is None:
            return None
        world = self._display_to_world(x, y)
        extent = self._image.GetExtent()
        axis = self.normal_axis()

        ijk = [0, 0, 0]
        u, v = self._world_to_slice_uv(world)
        out = self._reslice.GetOutput()
        u_idx = round((u - out.GetOrigin()[0]) / out.GetSpacing()[0])
        v_idx = round((v - out.GetOrigin()[1]) / out.GetSpacing()[1])

        # Map reslice output axes back to volume axes.
        mat = self._axes_matrix
        row_axes = []
        for row in range(2):
            vec = [mat.GetElement(row, c) for c in range(3)]
            ax = int(np.argmax(np.abs(vec)))
            row_axes.append(ax)
        ijk[row_axes[0]] = self._extent_offset(row_axes[0], u_idx, out, 0)
        ijk[row_axes[1]] = self._extent_offset(row_axes[1], v_idx, out, 1)
        ijk[axis] = extent[2 * axis] + self._slice_index

        for a in range(3):
            if ijk[a] < extent[2 * a] or ijk[a] > extent[2 * a + 1]:
                return None
        return (ijk[0], ijk[1], ijk[2])

    def _extent_offset(self, axis: int, out_idx: int,
                       out: vtkImageData, out_dim: int) -> int:
        """Convert reslice output index to volume index along *axis*."""
        ext = self._image.GetExtent()
        sp = self._image.GetSpacing()
        out_sp = out.GetSpacing()[out_dim]
        # Sign of the mapping.
        mat = self._axes_matrix
        vec = [mat.GetElement(out_dim, c) for c in range(3)]
        sign = 1.0 if vec[axis] > 0 else -1.0
        center = (ext[2 * axis] + ext[2 * axis + 1]) / 2.0
        half = out.GetExtent()[2 * out_dim + 1] / 2.0
        return int(round(center + sign * (out_idx - half)
                         * (out_sp / sp[axis])))

    # ================================================================== #
    # Overlay text / actor updates
    # ================================================================== #
    def _update_direction_markers(self) -> None:
        markers = DIRECTION_MARKERS[self._orientation]
        for key, actor in self._dir_actors.items():
            actor.SetInput(markers[key])

    def _update_info_text(self) -> None:
        axis = self.normal_axis()
        unit = ("X", "Y", "Z")[axis]
        pos = self._crosshair[axis] if self._image is not None else 0.0
        text = (f"{self._orientation.capitalize()}  "
                f"Slice: {self._slice_index + 1}/{self._num_slices}  "
                f"{unit}: {pos:.1f} mm  Zoom: {self._zoom_factor * 100:.0f}%")
        self._info_actor.SetInput(text)
        w, h = self._iren.GetRenderWindow().GetSize()
        self._info_actor.SetDisplayPosition(max(w - 380, 8), 8)

    def _update_crosshair_actor(self) -> None:
        if self._image is None:
            return
        u, v = self._world_to_slice_uv(self._crosshair)
        out = self._reslice.GetOutput()
        ext = out.GetExtent()
        u0, u1 = ext[0], ext[1]
        v0, v1 = ext[2], ext[3]
        mat = self._axes_matrix

        def to_world(pu, pv):
            p = mat.MultiplyPoint((pu, pv, 0.0, 1.0))
            return p[:3]

        p0, p1 = to_world(u0, v), to_world(u1, v)
        p2, p3 = to_world(u, v0), to_world(u, v1)
        pts = self._crosshair_points
        pts.SetPoint(0, *p0)
        pts.SetPoint(1, *p1)
        pts.SetPoint(2, *p2)
        pts.SetPoint(3, *p3)
        pts.Modified()

    # ================================================================== #
    # VTK interaction callbacks
    # ================================================================== #
    def _on_wheel_forward(self, _obj, _event) -> None:
        if self._iren.GetControlKey():
            self._zoom_by(0.9)
        else:
            self.set_slice_index(self._slice_index + 1)

    def _on_wheel_backward(self, _obj, _event) -> None:
        if self._iren.GetControlKey():
            self._zoom_by(1.1)
        else:
            self.set_slice_index(self._slice_index - 1)

    def _zoom_by(self, factor: float) -> None:
        scale = self._camera.GetParallelScale() * factor
        self._camera.SetParallelScale(scale)
        if hasattr(self, "_fit_parallel_scale") and self._fit_parallel_scale > 0:
            self._zoom_factor = self._fit_parallel_scale / scale
        self._update_info_text()
        self.zoom_changed.emit(self._zoom_factor)
        self.render()

    def zoom_factor(self) -> float:
        return self._zoom_factor

    def _on_left_press(self, _obj, _event) -> None:
        self._left_pressed = True
        self._last_pos = self._iren.GetEventPosition()
        self._press_pos = self._last_pos
        self._wl_start = self.get_window_level()

        # Label editor gets first refusal.
        editor = self._label_editor
        if editor is not None and editor.consume_left_button():
            editor.on_left_press(self._last_pos[0], self._last_pos[1])
            return
        self._iren.Render()

    def _on_middle_press(self, _obj, _event) -> None:
        self._middle_pressed = True
        self._last_pos = self._iren.GetEventPosition()

    def _on_mouse_move(self, _obj, _event) -> None:
        x, y = self._iren.GetEventPosition()
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]

        editor = self._label_editor
        if editor is not None and editor.consume_left_button() \
                and self._left_pressed and editor.stroke_active():
            editor.on_mouse_move(x, y)
        elif self._left_pressed:
            # Window / level: drag right = wider window, up = higher level.
            window, level = self._wl_start
            sensitivity = max(window, 1.0) / 300.0
            self.set_window_level(max(window + dx * sensitivity * 4, 1.0),
                                  level + dy * sensitivity * 4)
        elif self._middle_pressed:
            self._pan(dx, dy)

        self._last_pos = (x, y)

    def _pan(self, dx: float, dy: float) -> None:
        height = self._iren.GetRenderWindow().GetSize()[1] or 1
        world_per_px = (2.0 * self._camera.GetParallelScale()) / height
        fp = list(self._camera.GetFocalPoint())
        pos = list(self._camera.GetPosition())
        fp[0] -= dx * world_per_px
        fp[1] -= dy * world_per_px
        pos[0] -= dx * world_per_px
        pos[1] -= dy * world_per_px
        self._camera.SetFocalPoint(*fp)
        self._camera.SetPosition(*pos)
        self.render()

    def _on_left_release(self, _obj, _event) -> None:
        was_pressed = self._left_pressed
        self._left_pressed = False

        editor = self._label_editor
        if editor is not None and editor.consume_left_button():
            x, y = self._iren.GetEventPosition()
            editor.on_left_release(x, y)

        # A click without drag moves the crosshair (pointer behaviour).
        x, y = self._iren.GetEventPosition()
        moved = abs(x - self._press_pos[0]) + abs(y - self._press_pos[1])
        if was_pressed and moved < 4 and self._image is not None:
            world = self._display_to_world(x, y)
            # Snap the normal-axis component to the current plane.
            ch = list(world)
            ch[self.normal_axis()] = self._crosshair[self.normal_axis()] \
                if self._crosshair != (0.0, 0.0, 0.0) else world[self.normal_axis()]
            self.set_cursor(ch, emit=True)
            self.left_clicked.emit(tuple(ch))
        self.render()

    def _on_middle_release(self, _obj, _event) -> None:
        self._middle_pressed = False

    def _on_key_press(self, _obj, _event) -> None:
        key = self._iren.GetKeySym() or ""
        if key.lower() == "c":
            if self.cine_running():
                self.stop_cine()
            else:
                self.start_cine()
        elif key.lower() == "r":
            self.reset_view()
        elif key == "plus" or key == "equal":
            self._zoom_by(0.9)
        elif key == "minus":
            self._zoom_by(1.1)

    # ------------------------------------------------------------------ #
    def close(self) -> None:  # pragma: no cover - Qt lifecycle
        self.stop_cine()
        try:
            self._iren.close()
        except Exception:
            pass
        super().close()
