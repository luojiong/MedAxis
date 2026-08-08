"""SliceRenderer — orthographic MPR slice views (axial/coronal/sagittal)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.volume_data import VolumeData

try:
    import vtk

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    HAS_VTK = False

from .helpers import numpy_to_vtk_image, volume_user_matrix

logger = logging.getLogger(__name__)

__all__ = ["SliceRenderer", "AXIAL", "CORONAL", "SAGITTAL"]

AXIAL = "axial"
CORONAL = "coronal"
SAGITTAL = "sagittal"

_ORIENTATIONS = (AXIAL, CORONAL, SAGITTAL)

# Crosshair line colors per plane (spec: red=axial, green=coronal, blue=sagittal)
CROSSHAIR_COLORS = {
    AXIAL: (1.0, 0.0, 0.0),
    CORONAL: (0.0, 1.0, 0.0),
    SAGITTAL: (0.0, 0.0, 1.0),
}


def _require_vtk() -> None:
    if not HAS_VTK:
        raise RuntimeError("VTK is required for SliceRenderer")


class SliceRenderer:
    """Render one orthogonal slice of a volume in a ``vtkRenderer``.

    Uses ``vtkImageReslice`` with an orientation-dependent axes matrix so
    oblique volume directions are handled correctly. Window/level is
    applied through ``vtkImageMapToWindowLevelColors``.

    Typical usage::

        sr = SliceRenderer()
        sr.setup(renderer, volume, AXIAL, volume.array.shape[0] // 2)
        sr.set_window_level(350.0, 40.0)
    """

    def __init__(self) -> None:
        self.volume: Optional[VolumeData] = None
        self.orientation: str = AXIAL
        self.slice_index: int = 0

        self._image_data = None
        self._user_matrix = None
        self._reslice = None
        self._window_level_filter = None
        self._actor = None
        self._renderer = None
        self._crosshair_actors: dict[str, object] = {}
        self._crosshair_world: np.ndarray = np.zeros(3)
        self._window: float = 256.0
        self._level: float = 128.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, renderer, volume: VolumeData, orientation: str = AXIAL,
              slice_index: Optional[int] = None) -> None:
        """Configure the renderer to show one slice plane of the volume.

        Args:
            renderer: Target ``vtkRenderer``.
            volume: The volume to display.
            orientation: One of ``AXIAL``, ``CORONAL``, ``SAGITTAL``.
            slice_index: Initial slice index (defaults to the middle).
        """
        _require_vtk()
        if orientation not in _ORIENTATIONS:
            raise ValueError(f"orientation must be one of {_ORIENTATIONS}")

        self.volume = volume
        self.orientation = orientation
        self._renderer = renderer

        # Volume -> vtkImageData
        self._image_data = numpy_to_vtk_image(
            volume.array, volume.spacing_mm, volume.origin_mm
        )
        self._user_matrix = volume_user_matrix(volume)

        # Reslice pipeline
        self._reslice = vtk.vtkImageReslice()
        self._reslice.SetInputData(self._image_data)
        self._reslice.SetInterpolationModeToLinear()
        self._reslice.SetOutputDimensionality(2)
        self._reslice.SetAutoCropOutput(True)

        # Window/level
        self._window_level_filter = vtk.vtkImageMapToWindowLevelColors()
        self._window_level_filter.SetInputConnection(self._reslice.GetOutputPort())
        center, width = self._default_window(volume)
        self._window, self._level = width, center

        max_index = self.num_slices - 1
        self.slice_index = (slice_index if slice_index is not None
                            else max_index // 2)
        self.slice_index = max(0, min(self.slice_index, max_index))
        self._reslice.SetResliceAxes(self._axes_matrix())
        self._reslice.Update()

        # Actor
        self._actor = vtk.vtkImageActor()
        self._actor.GetMapper().SetInputConnection(self._window_level_filter.GetOutputPort())
        self._actor.SetUserMatrix(self._user_matrix)
        self._actor.SetInterpolate(True)
        self._window_level_filter.SetWindow(self._window)
        self._window_level_filter.SetLevel(self._level)

        renderer.AddActor(self._actor)
        renderer.ResetCamera()
        self._setup_parallel_camera(renderer)
        self._create_crosshair_actors(renderer)

    def teardown(self) -> None:
        """Remove all actors from the renderer."""
        if self._renderer is None:
            return
        if self._actor is not None:
            self._renderer.RemoveActor(self._actor)
        for actor in self._crosshair_actors.values():
            self._renderer.RemoveActor(actor)
        self._crosshair_actors.clear()
        self._actor = None
        self._renderer = None

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_slice(self, slice_index: int) -> None:
        """Move to a new slice index (clamped to valid range)."""
        if self._reslice is None:
            raise RuntimeError("SliceRenderer.setup() not called")
        self.slice_index = max(0, min(int(slice_index), self.num_slices - 1))
        self._reslice.SetResliceAxes(self._axes_matrix())
        self._reslice.Update()
        self._window_level_filter.Update()
        self._render()

    def update_orientation(self, orientation: str) -> None:
        """Switch the slice plane, keeping the slice index proportional."""
        if orientation not in _ORIENTATIONS:
            raise ValueError(f"orientation must be one of {_ORIENTATIONS}")
        if self.volume is None or orientation == self.orientation:
            return
        # Keep the relative position within the volume.
        fraction = (self.slice_index / max(self.num_slices - 1, 1)
                    if self.num_slices > 1 else 0.5)
        self.orientation = orientation
        new_index = int(round(fraction * (self.num_slices - 1)))
        self.update_slice(new_index)
        if self._renderer is not None:
            self._renderer.ResetCamera()
            self._setup_parallel_camera(self._renderer)

    def set_window_level(self, window: float, level: float) -> None:
        """Set the display window width / level (intensity units)."""
        if self._window_level_filter is None:
            raise RuntimeError("SliceRenderer.setup() not called")
        self._window = float(window)
        self._level = float(level)
        self._window_level_filter.SetWindow(self._window)
        self._window_level_filter.SetLevel(self._level)
        self._window_level_filter.Update()
        self._render()

    def get_window_level(self) -> tuple[float, float]:
        """Return the current ``(window, level)``."""
        return (self._window, self._level)

    # ------------------------------------------------------------------
    # Crosshair
    # ------------------------------------------------------------------
    def update_crosshair(self, world_position: tuple[float, float, float]) -> None:
        """Position the crosshair lines at a world coordinate (mm, LPS)."""
        self._crosshair_world = np.asarray(world_position, dtype=np.float64)
        if not self._crosshair_actors or self.volume is None:
            return

        bounds = self._plane_bounds()
        pos = self._crosshair_world
        for orientation, actor in self._crosshair_actors.items():
            p1, p2 = self._crosshair_segment(orientation, pos, bounds)
            src = actor.GetMapper().GetInputAlgorithm()  # vtkLineSource
            src.SetPoint1(*p1)
            src.SetPoint2(*p2)
            src.Update()
        self._render()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show/hide the crosshair line actors."""
        for actor in self._crosshair_actors.values():
            actor.SetVisibility(bool(visible))
        self._render()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def num_slices(self) -> int:
        """Number of slices along the current orientation."""
        if self.volume is None:
            return 0
        nk, nj, ni = self.volume.array.shape
        return {AXIAL: nk, CORONAL: nj, SAGITTAL: ni}[self.orientation]

    def get_reslice_axes(self, orientation: str, slice_index: int) -> np.ndarray:
        """Compute the 4x4 reslice element matrix for a plane.

        Rows are ``[x_axis | y_axis | z_axis(normal) | origin]`` in world
        (LPS) coordinates, consistent with the volume affine.
        """
        if self.volume is None:
            raise RuntimeError("SliceRenderer.setup() not called")
        affine = self.volume.affine
        a0, a1, a2 = affine[:3, 0], affine[:3, 1], affine[:3, 2]
        origin = affine[:3, 3]

        if orientation == AXIAL:          # fixed k
            x_axis, y_axis, normal = a0, a1, a2
            offset = slice_index * a2
        elif orientation == CORONAL:      # fixed j
            x_axis, y_axis, normal = a0, a2, a1
            offset = slice_index * a1
        elif orientation == SAGITTAL:     # fixed i
            x_axis, y_axis, normal = a1, a2, a0
            offset = slice_index * a0
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        element = np.eye(4, dtype=np.float64)
        element[:3, 0] = x_axis
        element[:3, 1] = y_axis
        element[:3, 2] = normal
        element[:3, 3] = origin + offset
        return element

    def get_slice_world_position(self) -> tuple[float, float, float]:
        """World position of the current slice center."""
        if self.volume is None:
            return (0.0, 0.0, 0.0)
        nk, nj, ni = self.volume.array.shape
        center = {
            AXIAL: (ni / 2.0, nj / 2.0, float(self.slice_index)),
            CORONAL: (ni / 2.0, float(self.slice_index), nk / 2.0),
            SAGITTAL: (float(self.slice_index), nj / 2.0, nk / 2.0),
        }[self.orientation]
        return self.volume.voxel_to_world(*center)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _axes_matrix(self):
        element = self.get_reslice_axes(self.orientation, self.slice_index)
        matrix = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                matrix.SetElement(r, c, float(element[r, c]))
        return matrix

    @staticmethod
    def _default_window(volume: VolumeData) -> tuple[float, float]:
        """Pick an initial (center, width) for the volume."""
        preset = getattr(volume, "default_window", None)
        if preset:
            return (float(preset[0]), float(preset[1]))
        if volume.modality.upper() == "CT":
            return (40.0, 350.0)  # soft tissue
        return volume.get_window_level_range()

    def _setup_parallel_camera(self, renderer) -> None:
        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        if self.volume is None:
            return
        center = np.asarray(self.get_slice_world_position())
        normal = self.get_reslice_axes(self.orientation, self.slice_index)[:3, 2]
        normal /= max(np.linalg.norm(normal), 1e-12)
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center - normal * 500.0))
        camera.SetViewUp(0.0, 0.0, 1.0)
        renderer.ResetCamera()

    def _create_crosshair_actors(self, renderer) -> None:
        for actor in self._crosshair_actors.values():
            renderer.RemoveActor(actor)
        self._crosshair_actors.clear()

        bounds = self._plane_bounds()
        pos = self._crosshair_world
        for orientation in (AXIAL, CORONAL, SAGITTAL):
            p1, p2 = self._crosshair_segment(orientation, pos, bounds)
            source = vtk.vtkLineSource()
            source.SetPoint1(*p1)
            source.SetPoint2(*p2)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*CROSSHAIR_COLORS[orientation])
            actor.GetProperty().SetLineWidth(1.0)
            actor.SetUserMatrix(self._user_matrix)
            renderer.AddActor(actor)
            self._crosshair_actors[orientation] = actor

    def _plane_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space (min, max) bounds of the current slice plane."""
        affine = self.volume.affine
        nk, nj, ni = self.volume.array.shape
        corners = np.array([
            [0, 0, 0], [ni, 0, 0], [0, nj, 0], [ni, nj, 0],
            [0, 0, nk], [ni, 0, nk], [0, nj, nk], [ni, nj, nk],
        ], dtype=np.float64)
        world = (affine[:3, :3] @ corners.T).T + affine[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    def _crosshair_segment(self, orientation: str, pos: np.ndarray,
                           bounds: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """A line through ``pos`` marking where another plane intersects.

        The line for plane ``orientation`` runs along that plane's
        in-plane axis perpendicular to the current view.
        """
        mn, mx = bounds
        if self.orientation == AXIAL:
            if orientation == SAGITTAL:   # blue vertical line (x = pos.x)
                return (np.array([pos[0], mn[1], pos[2]]), np.array([pos[0], mx[1], pos[2]]))
            if orientation == CORONAL:    # green horizontal line (y = pos.y)
                return (np.array([mn[0], pos[1], pos[2]]), np.array([mx[0], pos[1], pos[2]]))
            return (pos, pos)
        if self.orientation == CORONAL:
            if orientation == SAGITTAL:
                return (np.array([pos[0], pos[1], mn[2]]), np.array([pos[0], pos[1], mx[2]]))
            if orientation == AXIAL:
                return (np.array([mn[0], pos[1], pos[2]]), np.array([mx[0], pos[1], pos[2]]))
            return (pos, pos)
        # SAGITTAL view
        if orientation == CORONAL:
            return (np.array([pos[0], mn[1], pos[2]]), np.array([pos[0], mx[1], pos[2]]))
        if orientation == AXIAL:
            return (np.array([pos[0], pos[1], mn[2]]), np.array([pos[0], pos[1], mx[2]]))
        return (pos, pos)

    def _render(self) -> None:
        if self._renderer is not None:
            window = self._renderer.GetRenderWindow()
            if window is not None:
                window.Render()
