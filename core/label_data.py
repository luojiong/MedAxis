"""LabelData — segmentation label map data model."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from .units import mm3_to_ml

__all__ = ["LabelData", "LabelStats"]

# Default RGBA colors for auto-assigned label classes (0..N).
_DEFAULT_PALETTE: tuple[tuple[int, int, int, int], ...] = (
    (230, 25, 75, 255),
    (60, 180, 75, 255),
    (255, 225, 25, 255),
    (0, 130, 200, 255),
    (245, 130, 48, 255),
    (145, 30, 180, 255),
    (70, 240, 240, 255),
    (240, 50, 230, 255),
    (210, 245, 60, 255),
    (250, 190, 212, 255),
)


@dataclass
class LabelStats:
    """Quantitative measurements of a segmentation label."""

    label_id: int = 0
    voxel_count: int = 0
    volume_mm3: float = 0.0
    volume_ml: float = 0.0
    surface_area_mm2: float = 0.0
    mean_hu: float = 0.0
    std_hu: float = 0.0
    min_hu: float = 0.0
    max_hu: float = 0.0
    max_3d_diameter_mm: float = 0.0
    num_components: int = 0
    centroid_world: Optional[tuple[float, float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "label_id": self.label_id,
            "voxel_count": self.voxel_count,
            "volume_mm3": self.volume_mm3,
            "volume_ml": self.volume_ml,
            "surface_area_mm2": self.surface_area_mm2,
            "mean_hu": self.mean_hu,
            "std_hu": self.std_hu,
            "min_hu": self.min_hu,
            "max_hu": self.max_hu,
            "max_3d_diameter_mm": self.max_3d_diameter_mm,
            "num_components": self.num_components,
            "centroid_world": self.centroid_world,
        }


@dataclass
class LabelData:
    """A segmentation label map attached to a volume.

    Attributes:
        id: Unique label identifier within the project.
        name: Display name.
        parent_volume_id: ID of the volume this label map overlays.
        array: 3D int16 label map. 0 = background; positive integers are
            class indices (matching :attr:`class_names` keys).
        color: Display color as RGBA (0-255).
        opacity: Display opacity in [0, 1].
        visible: Whether the label is shown in viewers.
        class_names: Mapping of label value -> human-readable name.
        class_colors: Mapping of label value -> RGBA color.
        source: Provenance, e.g. "manual", "algorithm:threshold",
            "ai:totalsegmentator", "ai:sam".
        mesh: Optional reference to a derived surface mesh
            (:class:`~core.mesh_data.MeshData` or its id).
    """

    id: str
    name: str
    parent_volume_id: str
    array: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.int16))
    color: tuple[int, int, int, int] = (230, 25, 75, 255)
    opacity: float = 0.5
    visible: bool = True
    class_names: dict[int, str] = field(default_factory=dict)
    class_colors: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    source: str = "manual"
    mesh: Any = None

    def __post_init__(self) -> None:
        if self.array.dtype != np.int16:
            self.array = self.array.astype(np.int16)
        # Ensure every class present in the array has a name/color.
        for value in np.unique(self.array):
            if value == 0:
                continue
            self.class_names.setdefault(int(value), f"Label {value}")
            if int(value) not in self.class_colors:
                self.class_colors[int(value)] = _DEFAULT_PALETTE[int(value) % len(_DEFAULT_PALETTE)]

    @property
    def data(self) -> np.ndarray:
        """Compatibility alias used by interactive editing widgets."""
        return self.array

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self.array = np.asarray(value, dtype=np.int16)

    def refresh(self) -> None:
        """Hook for view models; label arrays are immediately mutable."""
        return

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(self, volume=None, label_value: Optional[int] = None) -> LabelStats:
        """Compute quantitative measurements for this label map.

        Args:
            volume: Optional parent :class:`~core.volume_data.VolumeData`
                used for Hounsfield-unit statistics.
            label_value: Restrict stats to one class value. Defaults to all
                non-zero labels combined.

        Returns:
            A :class:`LabelStats` instance.
        """
        mask = (self.array != 0) if label_value is None else (self.array == label_value)
        voxel_count = int(mask.sum())
        stats = LabelStats(label_id=label_value or 0, voxel_count=voxel_count)
        if voxel_count == 0:
            return stats

        # Voxel spacing (mm) from parent volume, else isotropic 1 mm.
        spacing = (1.0, 1.0, 1.0)
        if volume is not None:
            spacing = tuple(volume.spacing_mm)
        voxel_vol = spacing[0] * spacing[1] * spacing[2]

        stats.volume_mm3 = voxel_count * voxel_vol
        stats.volume_ml = mm3_to_ml(stats.volume_mm3)

        # Surface area: use marching cubes if available, else face counting.
        stats.surface_area_mm2 = self._surface_area(mask, spacing)

        # Connected components via scipy if available.
        stats.num_components = self._count_components(mask)

        # Max 3D diameter: bounding-box diagonal of the mask (approximation).
        coords = np.argwhere(mask)  # (N, 3) in (k, j, i)
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        diag = (mx - mn) * np.array([spacing[2], spacing[1], spacing[0]])
        stats.max_3d_diameter_mm = float(np.linalg.norm(diag))

        # Centroid in world coordinates if a volume is provided.
        if volume is not None:
            centroid_vox = coords.mean(axis=0)  # (k, j, i)
            stats.centroid_world = volume.voxel_to_world(
                float(centroid_vox[2]), float(centroid_vox[1]), float(centroid_vox[0])
            )
            # HU statistics from the parent volume.
            hu = volume.array[mask]
            stats.mean_hu = float(hu.mean())
            stats.std_hu = float(hu.std())
            stats.min_hu = float(hu.min())
            stats.max_hu = float(hu.max())

        return stats

    @staticmethod
    def _surface_area(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
        """Estimate surface area (mm^2) of a binary mask."""
        try:
            from skimage.measure import marching_cubes, mesh_surface_area  # type: ignore

            verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
            return float(mesh_surface_area(verts, faces))
        except Exception:
            # Fallback: count exposed faces between mask and background.
            area = 0.0
            face_areas = (
                spacing[0] * spacing[1],  # faces perpendicular to z (axis 0)
                spacing[0] * spacing[2],  # faces perpendicular to y (axis 1)
                spacing[1] * spacing[2],  # faces perpendicular to x (axis 2)
            )
            for axis in range(3):
                diff = np.diff(mask.astype(np.int8), axis=axis, prepend=0, append=0)
                area += float(np.count_nonzero(diff)) * face_areas[axis]
            return area

    @staticmethod
    def _count_components(mask: np.ndarray) -> int:
        """Count connected components (26-connectivity) of a binary mask."""
        try:
            from scipy import ndimage  # type: ignore

            structure = np.ones((3, 3, 3), dtype=np.int8)
            _, n = ndimage.label(mask, structure=structure)
            return int(n)
        except ImportError:
            return 1  # conservative fallback

    # ------------------------------------------------------------------
    # Contours
    # ------------------------------------------------------------------

    def get_contour(
        self, slice_index: int, axis: int = 0, label_value: Optional[int] = None
    ) -> list[np.ndarray]:
        """Extract contour polylines for one 2D slice.

        Args:
            slice_index: Index along ``axis``.
            axis: 0 (axial), 1 (coronal), 2 (sagittal).
            label_value: Restrict to one class; defaults to all non-zero.

        Returns:
            List of contour arrays, each shaped (N, 2) in slice-pixel
            coordinates. Empty list if the slice has no label pixels.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        slab = np.take(self.array, slice_index, axis=axis)
        mask = (slab != 0) if label_value is None else (slab == label_value)
        if not mask.any():
            return []

        try:
            from skimage.measure import find_contours  # type: ignore

            return [c for c in find_contours(mask.astype(np.float32), level=0.5)]
        except ImportError:
            # Fallback: return boundary pixels of each connected blob as
            # single point clouds (no ordering guarantee).
            contours: list[np.ndarray] = []
            interior = (
                mask
                & np.roll(mask, 1, 0)
                & np.roll(mask, -1, 0)
                & np.roll(mask, 1, 1)
                & np.roll(mask, -1, 1)
            )
            boundary = mask & ~interior
            pts = np.argwhere(boundary)
            if len(pts):
                contours.append(pts.astype(np.float32))
            return contours

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (not the array) to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "parent_volume_id": self.parent_volume_id,
            "shape": tuple(self.array.shape),
            "color": tuple(self.color),
            "opacity": self.opacity,
            "visible": self.visible,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "class_colors": {str(k): tuple(c) for k, c in self.class_colors.items()},
            "source": self.source,
        }
