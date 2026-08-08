"""VolumeData — 3D image volume data model."""
from __future__ import annotations

from functools import cached_property
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from .image_data import ImageData
from .transform import build_affine_matrix, world_to_voxel_coords, voxel_to_world_coords

__all__ = ["VolumeData"]

# Axis indices
AXIAL, CORONAL, SAGITTAL = 0, 1, 2


@dataclass
class VolumeData:
    """A 3D image volume with physical-space metadata.

    Attributes:
        array: 3D numpy array indexed as ``array[k, j, i]`` (z, y, x) in
            voxel space, matching ITK/NumPy conventions.
        spacing: Voxel size (sx, sy, sz) in mm.
        origin: World coordinate (mm) of voxel (0, 0, 0) in LPS.
        direction: 9-element flat direction-cosine matrix (row-major),
            ITK convention.
        name: Human-readable volume name.
        modality: DICOM modality code, e.g. "CT", "MR".
    """

    array: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    name: str = "Volume"
    modality: str = ""

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"VolumeData requires a 3D array, got {self.array.ndim}D")

    # ------------------------------------------------------------------
    # Geometry properties
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Volume dimensions (ni, nj, nk) along x, y, z voxel axes."""
        nk, nj, ni = self.array.shape
        return (ni, nj, nk)

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        """Voxel spacing (sx, sy, sz) in millimetres."""
        return tuple(float(s) for s in self.spacing)  # type: ignore[return-value]

    @property
    def origin_mm(self) -> tuple[float, float, float]:
        """World origin (mm) of voxel (0, 0, 0)."""
        return tuple(float(o) for o in self.origin)  # type: ignore[return-value]

    @property
    def direction_matrix(self) -> np.ndarray:
        """3x3 direction cosine matrix."""
        return np.asarray(self.direction, dtype=np.float64).reshape(3, 3)

    @property
    def affine(self) -> np.ndarray:
        """4x4 voxel->world affine matrix."""
        return build_affine_matrix(self.spacing, self.origin, self.direction)

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Extract a 2D slice along the given voxel axis.

        Args:
            axis: 0 (axial / z), 1 (coronal / y), or 2 (sagittal / x).
            index: Slice index along that axis.

        Returns:
            2D numpy array view of the slice.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        if not (0 <= index < self.array.shape[axis]):
            raise IndexError(
                f"Slice index {index} out of range for axis {axis} "
                f"(size {self.array.shape[axis]})"
            )
        return np.take(self.array, index, axis=axis)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_voxel(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Convert a world point (mm, LPS) to continuous voxel indices (i, j, k)."""
        ijk = world_to_voxel_coords(np.array([x, y, z]), self.affine)
        return (float(ijk[0]), float(ijk[1]), float(ijk[2]))

    def voxel_to_world(self, i: float, j: float, k: float) -> tuple[float, float, float]:
        """Convert voxel indices (i, j, k) to world coordinates (mm, LPS)."""
        xyz = voxel_to_world_coords(np.array([i, j, k]), self.affine)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    # ------------------------------------------------------------------
    # Statistics / windowing
    # ------------------------------------------------------------------

    def histogram(self, bins: int = 256, range: Optional[tuple[float, float]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute the intensity histogram of the volume.

        Args:
            bins: Number of histogram bins.
            range: Optional (min, max) range; defaults to data range.

        Returns:
            (counts, bin_edges) tuple.
        """
        return np.histogram(self.array, bins=bins, range=range)

    def get_min_max(self) -> tuple[float, float]:
        """Return (min, max) intensity of the volume."""
        return (float(self.array.min()), float(self.array.max()))

    def get_window_level_range(self, percentile_low: float = 1.0, percentile_high: float = 99.0) -> tuple[float, float]:
        """Suggest a display window (level, width) based on percentiles.

        Robust to outlier voxels (e.g. metal artifacts in CT).

        Returns:
            (window_center, window_width) in intensity units.
        """
        lo, hi = np.percentile(self.array, [percentile_low, percentile_high])
        center = float((lo + hi) / 2.0)
        width = float(hi - lo)
        return (center, max(width, 1.0))

    # ------------------------------------------------------------------
    # Construction / export
    # ------------------------------------------------------------------

    @classmethod
    def from_image_data_list(cls, slices: list[ImageData], name: str = "Volume") -> "VolumeData":
        """Build a volume from a list of 2D :class:`ImageData` slices.

        Slices are sorted along the scan direction (by slice location, or
        by projection of the origin onto the slice normal) and stacked.

        Args:
            slices: Non-empty list of slices sharing one series.
            name: Name for the resulting volume.

        Returns:
            A new :class:`VolumeData`.
        """
        if not slices:
            raise ValueError("Cannot build a volume from an empty slice list")
        if len(slices) == 1:
            s = slices[0]
            arr = s.array[np.newaxis, :, :]  # (1, rows, cols)
            row_cos = np.array(s.orientation[:3])
            col_cos = np.array(s.orientation[3:6])
            normal = np.cross(row_cos, col_cos)
            direction = tuple(
                float(v) for v in np.column_stack([row_cos, col_cos, normal]).reshape(-1)
            )
            return cls(
                array=arr,
                spacing=(float(s.spacing[1]), float(s.spacing[0]), 1.0),
                origin=s.origin,
                direction=direction,
                name=name,
            )

        ref = slices[0]
        row_cos, col_cos = np.array(ref.orientation[:3]), np.array(ref.orientation[3:6])
        normal = np.cross(row_cos, col_cos)

        def sort_key(sl: ImageData) -> float:
            if sl.slice_location is not None:
                return sl.slice_location
            return float(np.dot(np.asarray(sl.origin), normal))

        ordered = sorted(slices, key=sort_key)

        # Slice thickness from consecutive origins (fallback 1.0 mm).
        if len(ordered) >= 2:
            d = np.asarray(ordered[1].origin) - np.asarray(ordered[0].origin)
            slice_spacing = float(np.linalg.norm(d)) or 1.0
        else:
            slice_spacing = 1.0

        stack = np.stack([sl.array for sl in ordered], axis=0)  # (n, rows, cols)
        origin = tuple(float(v) for v in ordered[0].origin)  # type: ignore[assignment]

        # ITK spacing order: (x=col, y=row, z=slice)
        spacing = (float(ref.spacing[1]), float(ref.spacing[0]), slice_spacing)

        direction = tuple(float(v) for v in np.column_stack([row_cos, col_cos, normal]).reshape(-1))

        return cls(
            array=stack,
            spacing=spacing,  # type: ignore[arg-type]
            origin=origin,  # type: ignore[arg-type]
            direction=direction,
            name=name,
            modality=getattr(ref, "modality", ""),
        )

    def to_numpy(self) -> np.ndarray:
        """Return the underlying 3D numpy array."""
        return self.array

    @cached_property
    def image_data(self):
        """A ``vtkImageData`` sharing this volume's voxel memory (zero-copy).

        The numpy buffer is shared with VTK when the array is C-contiguous,
        so renderers do not copy the volume.  The direction cosine matrix is
        *not* applied here: renderers apply it via ``SetUserMatrix``
        (:func:`rendering.helpers.volume_user_matrix`) to avoid double
        transforms on actors that already carry a user matrix.
        """
        import vtk
        from vtkmodules.util.numpy_support import numpy_to_vtk

        array = np.ascontiguousarray(self.array)
        vtk_data = numpy_to_vtk(array.ravel(), deep=False)
        vtk_data.SetNumberOfComponents(1)

        nk, nj, ni = array.shape
        image = vtk.vtkImageData()
        image.SetDimensions(ni, nj, nk)
        image.SetSpacing(float(self.spacing[0]), float(self.spacing[1]), float(self.spacing[2]))
        image.SetOrigin(float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        image.GetPointData().SetScalars(vtk_data)
        return image

    def to_itk_image(self):
        """Convert to an ``itk.Image`` (requires the ``itk`` package).

        Spacing, origin and direction are preserved.
        """
        try:
            import itk  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("The 'itk' package is required for to_itk_image()") from exc

        image = itk.GetImageFromArray(self.array)
        image.SetSpacing([float(s) for s in self.spacing])
        image.SetOrigin([float(o) for o in self.origin])
        # SetDirection accepts an itkMatrix or a 2D numpy array, not a flat list.
        image.SetDirection(np.asarray(self.direction, dtype=float).reshape(3, 3))
        return image

    def get_bridge_handle(self) -> dict[str, Any]:
        """Return an opaque handle for DataBridge inter-process sharing.

        The handle contains everything needed to reconstruct or share the
        volume without copying pixel data eagerly.
        """
        return {
            "type": "VolumeData",
            "name": self.name,
            "shape": tuple(self.array.shape),
            "dtype": str(self.array.dtype),
            "spacing": self.spacing_mm,
            "origin": self.origin_mm,
            "direction": tuple(self.direction),
            "buffer": self.array,  # shared reference; DataBridge may wrap this
        }
