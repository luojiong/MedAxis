"""MPRRenderer — arbitrary oblique multi-planar reformation."""
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

__all__ = ["MPRRenderer"]


def _require_vtk() -> None:
    if not HAS_VTK:
        raise RuntimeError("VTK is required for MPRRenderer")


class MPRRenderer:
    """Render an arbitrary oblique plane through a volume.

    The plane is defined by a world-space origin and normal; a slab of
    configurable thickness can be collapsed via mean/MIP projection.
    An interactive plane widget is attached when an interactor is given.
    """

    def __init__(self) -> None:
        self.volume: Optional[VolumeData] = None
        self._renderer = None
        self._image_data = None
        self._reslice = None
        self._actor = None
        self._widget = None

        self._origin: np.ndarray = np.zeros(3)
        self._normal: np.ndarray = np.array([0.0, 0.0, 1.0])
        self._slab_thickness_mm: float = 0.0
        self._slab_mode: str = "mean"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, renderer, volume: VolumeData,
              interactor=None,
              origin: Optional[tuple[float, float, float]] = None,
              normal: Optional[tuple[float, float, float]] = None) -> None:
        """Configure the renderer with an interactive oblique plane.

        Args:
            renderer: Target ``vtkRenderer``.
            volume: Volume to reslice.
            interactor: Optional interactor to enable the plane widget.
            origin: Plane origin (mm). Defaults to the volume center.
            normal: Plane normal. Defaults to (0, 0, 1).
        """
        _require_vtk()
        self.volume = volume
        self._renderer = renderer

        self._image_data = numpy_to_vtk_image(
            volume.array, volume.spacing_mm, volume.origin_mm
        )

        self._reslice = vtk.vtkImageReslice()
        self._reslice.SetInputData(self._image_data)
        self._reslice.SetInterpolationModeToLinear()
        self._reslice.SetOutputDimensionality(2)
        self._reslice.SetAutoCropOutput(True)

        if origin is None:
            nk, nj, ni = volume.array.shape
            origin = volume.voxel_to_world(ni / 2.0, nj / 2.0, nk / 2.0)
        if normal is None:
            normal = tuple(float(v) for v in volume.direction_matrix[:, 2])
        self._origin = np.asarray(origin, dtype=np.float64)
        self._normal = self._normalize(np.asarray(normal, dtype=np.float64))
        self._reslice.SetResliceAxes(self._plane_matrix())
        self._reslice.Update()

        self._actor = vtk.vtkImageActor()
        self._actor.GetMapper().SetInputConnection(self._reslice.GetOutputPort())
        self._actor.SetUserMatrix(volume_user_matrix(volume))
        self._actor.SetInterpolate(True)
        renderer.AddActor(self._actor)
        renderer.ResetCamera()

        if interactor is not None:
            self._setup_widget(interactor)

    def teardown(self) -> None:
        """Remove actors and disable the widget."""
        if self._widget is not None:
            self._widget.SetEnabled(False)
            self._widget = None
        if self._renderer is not None and self._actor is not None:
            self._renderer.RemoveActor(self._actor)
        self._renderer = None
        self._actor = None

    # ------------------------------------------------------------------
    # Plane control
    # ------------------------------------------------------------------
    def set_reslice_plane(self, origin: tuple[float, float, float],
                          normal: tuple[float, float, float]) -> None:
        """Update the plane position/orientation and refresh the view."""
        self._origin = np.asarray(origin, dtype=np.float64)
        self._normal = self._normalize(np.asarray(normal, dtype=np.float64))
        if self._reslice is None:
            raise RuntimeError("MPRRenderer.setup() not called")
        self._reslice.SetResliceAxes(self._plane_matrix())
        self._reslice.Update()
        self._render()

    def translate_plane(self, offset_mm: float) -> None:
        """Translate the plane along its normal by ``offset_mm``."""
        new_origin = self._origin + self._normal * float(offset_mm)
        self.set_reslice_plane(tuple(new_origin), tuple(self._normal))

    def rotate_plane(self, axis: tuple[float, float, float], angle_deg: float) -> None:
        """Rotate the plane normal about ``axis`` by ``angle_deg``."""
        axis = self._normalize(np.asarray(axis, dtype=np.float64))
        angle = np.radians(angle_deg)
        # Rodrigues' rotation formula.
        k = axis
        n = self._normal
        rotated = (n * np.cos(angle) + np.cross(k, n) * np.sin(angle)
                   + k * np.dot(k, n) * (1.0 - np.cos(angle)))
        self.set_reslice_plane(tuple(self._origin), tuple(rotated))

    def get_current_plane(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the current ``(origin, normal)``."""
        return (tuple(float(v) for v in self._origin),  # type: ignore[return-value]
                tuple(float(v) for v in self._normal))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Slab mode
    # ------------------------------------------------------------------
    def set_slab_thickness(self, mm: float, mode: str = "mean") -> None:
        """Set slab projection thickness in mm.

        Args:
            mm: Slab thickness; 0 disables slabbing (single slice).
            mode: ``"mean"``, ``"max"`` (MIP), ``"min"`` or ``"sum"``.
        """
        if self._reslice is None:
            raise RuntimeError("MPRRenderer.setup() not called")
        self._slab_thickness_mm = max(0.0, float(mm))
        self._slab_mode = mode

        if self._slab_thickness_mm <= 0.0:
            self._reslice.SetSlabNumberOfSlices(1)
            self._render()
            return

        mode_map = {
            "mean": getattr(vtk, "VTK_IMAGE_SLAB_MODE_MEAN", 1),
            "max": getattr(vtk, "VTK_IMAGE_SLAB_MODE_MAX", 2),
            "mip": getattr(vtk, "VTK_IMAGE_SLAB_MODE_MAX", 2),
            "min": getattr(vtk, "VTK_IMAGE_SLAB_MODE_MIN", 0),
            "sum": getattr(vtk, "VTK_IMAGE_SLAB_MODE_SUM", 3),
        }
        slab_mode = mode_map.get(mode.lower(), mode_map["mean"])
        try:
            self._reslice.SetSlabMode(slab_mode)
        except AttributeError:
            logger.debug("vtkImageReslice.SetSlabMode unavailable on this VTK build")

        slice_spacing = min(self.volume.spacing_mm) if self.volume else 1.0
        num_slices = max(1, int(round(self._slab_thickness_mm / max(slice_spacing, 1e-6))))
        self._reslice.SetSlabNumberOfSlices(num_slices)
        self._reslice.Update()
        self._render()

    @property
    def slab_thickness(self) -> float:
        return self._slab_thickness_mm

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(vec)
        if length < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return vec / length

    def _plane_matrix(self):
        """4x4 reslice element matrix for the current plane."""
        normal = self._normal
        # Pick a reference vector not parallel to the normal.
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(normal, ref))) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        x_axis = self._normalize(np.cross(ref, normal))
        y_axis = self._normalize(np.cross(normal, x_axis))

        element = np.eye(4, dtype=np.float64)
        element[:3, 0] = x_axis
        element[:3, 1] = y_axis
        element[:3, 2] = normal
        element[:3, 3] = self._origin

        matrix = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                matrix.SetElement(r, c, float(element[r, c]))
        return matrix

    def _setup_widget(self, interactor) -> None:
        """Attach an interactive plane widget (vtkImagePlaneWidget)."""
        try:
            widget = vtk.vtkImagePlaneWidget()
        except AttributeError:
            logger.warning("vtkImagePlaneWidget unavailable in this VTK build")
            return
        widget.SetInteractor(interactor)
        widget.SetInputData(self._image_data)
        widget.SetPlaneOrientationToZAxes()
        nk = self._image_data.GetDimensions()[2]
        widget.SetSliceIndex(nk // 2)
        widget.SetResliceInterpolateToLinear()
        widget.SetWindowLevel(*self._auto_window_level())
        widget.GetPlaneProperty().SetColor(0.9, 0.7, 0.0)
        widget.On()
        widget.AddObserver("InteractionEvent", self._on_widget_interaction)
        self._widget = widget

    def _on_widget_interaction(self, widget, _event: str) -> None:
        """Sync internal plane state from the widget."""
        origin = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        widget.GetOrigin(origin)
        widget.GetNormal(normal)
        self._origin = np.asarray(origin, dtype=np.float64)
        self._normal = self._normalize(np.asarray(normal, dtype=np.float64))

    def _auto_window_level(self) -> tuple[float, float]:
        if self.volume is None:
            return (256.0, 128.0)
        center, width = self.volume.get_window_level_range()
        return (width, center)

    def _render(self) -> None:
        if self._renderer is not None:
            window = self._renderer.GetRenderWindow()
            if window is not None:
                window.Render()
