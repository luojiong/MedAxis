"""CPRRenderer — curved planar reformation along a B-spline path."""
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

from geometry.occ_wrapper import BSplineCurve

logger = logging.getLogger(__name__)

__all__ = ["CPRRenderer", "MODE_STRETCH", "MODE_STRAIGHTEN"]

MODE_STRETCH = "stretch"
MODE_STRAIGHTEN = "straighten"

try:
    from scipy.ndimage import map_coordinates

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    map_coordinates = None  # type: ignore[assignment]
    HAS_SCIPY = False


class CPRRenderer:
    """Curved planar reformation along a B-spline curve.

    The curve is fitted through user control points via the OCC-backed
    :class:`geometry.occ_wrapper.BSplineCurve`. Resampling the volume
    along the curve produces a straightened 2D strip (e.g. for vessels,
    ureters or the aorta).

    Display modes:
        - ``straighten``: samples at uniform arc length along the curve
          (structure appears straight, true length preserved).
        - ``stretch``: samples along the straight chord between the curve
          endpoints (structure is stretched onto a fixed-length axis).
    """

    def __init__(self) -> None:
        self.volume: Optional[VolumeData] = None
        self._renderer = None
        self._actor = None
        self._curve: Optional[BSplineCurve] = None
        self._sampled_points: Optional[np.ndarray] = None
        self._tangents: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._thickness_mm: float = 10.0
        self._mode: str = MODE_STRAIGHTEN
        self._num_samples: int = 256
        self._strip: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, renderer, volume: VolumeData) -> None:
        """Attach the CPR renderer to a ``vtkRenderer``."""
        self.volume = volume
        self._renderer = renderer

    def teardown(self) -> None:
        """Remove the strip actor from the renderer."""
        if self._renderer is not None and self._actor is not None:
            self._renderer.RemoveActor(self._actor)
        self._renderer = None
        self._actor = None

    # ------------------------------------------------------------------
    # Curve control
    # ------------------------------------------------------------------
    def set_curve_control_points(self, points: np.ndarray,
                                 num_samples: int = 256) -> BSplineCurve:
        """Fit a B-spline through control points and sample it.

        Args:
            points: (N, 3) world coordinates (mm, LPS), N >= 2.
            num_samples: Number of samples along the curve.

        Returns:
            The fitted :class:`BSplineCurve`.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if points.shape[0] < 2:
            raise ValueError("At least two control points are required")

        self._num_samples = max(2, int(num_samples))
        self._curve = BSplineCurve(points)
        self._sampled_points, self._tangents, self._normals = (
            self._curve.sample_equal_arc_length(self._num_samples)
        )
        return self._curve

    def set_thickness(self, mm: float) -> None:
        """Set the strip half-width in mm (total width = 2 * thickness)."""
        self._thickness_mm = max(0.1, float(mm))

    def set_display_mode(self, mode: str) -> None:
        """Set display mode: ``"straighten"`` or ``"stretch"``."""
        if mode not in (MODE_STRETCH, MODE_STRAIGHTEN):
            raise ValueError(f"mode must be '{MODE_STRETCH}' or '{MODE_STRAIGHTEN}'")
        self._mode = mode

    # ------------------------------------------------------------------
    # Reslicing
    # ------------------------------------------------------------------
    def reslice_along_curve(self) -> np.ndarray:
        """Resample the volume into a straightened 2D image.

        Returns:
            2D float32 array of shape ``(num_samples, num_offsets)`` —
            rows run along the curve, columns across the strip width.

        Raises:
            RuntimeError: If no curve has been set.
        """
        if self.volume is None:
            raise RuntimeError("CPRRenderer.setup() not called")
        if self._sampled_points is None:
            raise RuntimeError("Call set_curve_control_points() first")

        points, tangents, normals = self._sample_locations()
        spacing_x = self._cross_spacing()
        half = self._thickness_mm
        num_offsets = max(2, int(round(2.0 * half / spacing_x)) + 1)
        offsets = np.linspace(-half, half, num_offsets)

        # (S, O, 3) world sampling grid
        grid = points[:, np.newaxis, :] + offsets[np.newaxis, :, np.newaxis] * normals[:, np.newaxis, :]
        flat = grid.reshape(-1, 3)

        values = self._sample_world(flat)
        strip = values.reshape(points.shape[0], num_offsets).astype(np.float32)
        self._strip = strip
        return strip

    def display_strip(self) -> Optional[object]:
        """Create/update a ``vtkImageActor`` showing the CPR strip."""
        if not HAS_VTK:
            raise RuntimeError("VTK is required for display_strip()")
        if self._strip is None:
            self.reslice_along_curve()
        from .helpers import numpy_to_vtk_image

        strip = self._strip
        spacing_x = self._cross_spacing()
        arc_step = self._arc_length() / max(strip.shape[0] - 1, 1)
        image = numpy_to_vtk_image(strip[np.newaxis, :, :], (spacing_x, arc_step, 1.0), (0, 0, 0))

        if self._actor is None:
            self._actor = vtk.vtkImageActor()
            self._actor.GetMapper().SetInputData(image)
            if self._renderer is not None:
                self._renderer.AddActor(self._actor)
                self._renderer.ResetCamera()
        else:
            self._actor.GetMapper().SetInputData(image)
        self._actor._medaxis_image = image  # keep alive
        self._render()
        return self._actor

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sample_locations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample points/tangents/normals for the current display mode."""
        if self._mode == MODE_STRAIGHTEN:
            return self._sampled_points, self._tangents, self._normals

        # Stretch mode: positions along the chord, normals from nearest
        # curve point.
        start, end = self._sampled_points[0], self._sampled_points[-1]
        n = self._num_samples
        chord_points = start[np.newaxis, :] + (
            np.linspace(0.0, 1.0, n)[:, np.newaxis] * (end - start)[np.newaxis, :]
        )
        # Nearest curve point per chord position.
        dists = np.linalg.norm(
            chord_points[:, np.newaxis, :] - self._sampled_points[np.newaxis, :, :], axis=2
        )
        nearest = dists.argmin(axis=1)
        return chord_points, self._tangents[nearest], self._normals[nearest]

    def _sample_world(self, world_points: np.ndarray) -> np.ndarray:
        """Trilinearly sample the volume at world coordinates."""
        volume = self.volume
        affine = volume.affine
        inv = np.linalg.inv(affine)
        homog = np.column_stack([world_points, np.ones(world_points.shape[0])])
        ijk = (inv @ homog.T).T[:, :3]  # (i, j, k)

        nk, nj, ni = volume.array.shape
        coords_kji = np.column_stack([ijk[:, 2], ijk[:, 1], ijk[:, 0]])

        if HAS_SCIPY:
            return map_coordinates(volume.array, coords_kji.T, order=1,
                                   mode="constant", cval=float(volume.array.min()))
        # Nearest-neighbor fallback.
        idx = np.round(coords_kji).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, nk - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, nj - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, ni - 1)
        return volume.array[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)

    def _cross_spacing(self) -> float:
        """Smallest in-plane voxel size (mm) for across-curve sampling."""
        if self.volume is None:
            return 1.0
        return max(min(self.volume.spacing_mm), 1e-3)

    def _arc_length(self) -> float:
        """Total arc length of the sampled curve (mm)."""
        if self._sampled_points is None:
            return 0.0
        return float(np.linalg.norm(np.diff(self._sampled_points, axis=0), axis=1).sum())

    def _render(self) -> None:
        if self._renderer is not None:
            window = self._renderer.GetRenderWindow()
            if window is not None:
                window.Render()
