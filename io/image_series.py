"""Series assembly — group, sort, validate and stack DICOM slices."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.image_data import ImageData
from core.volume_data import VolumeData

logger = logging.getLogger(__name__)

__all__ = ["SeriesBuilder"]

_ORIENTATION_TOLERANCE = 0.999      # cos(angle) between normals
_SPACING_TOLERANCE = 1e-3           # mm


class SeriesBuilder:
    """Group DICOM instances into series and assemble them into volumes.

    Handles both classic single-frame stacks and multi-frame (enhanced)
    DICOM objects, which are expanded into per-frame slices before
    sorting and stacking.
    """

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------
    def build_series(self, images: list[ImageData]) -> dict[str, list[ImageData]]:
        """Group images by SeriesInstanceUID.

        Multi-frame instances are expanded so every frame becomes its own
        :class:`ImageData` entry with a derived origin.

        Returns:
            Mapping ``series_uid -> list of slices``.
        """
        expanded = self._expand_multiframe(images)
        grouped: dict[str, list[ImageData]] = {}
        for image in expanded:
            grouped.setdefault(image.series_uid or "<no-series-uid>", []).append(image)
        for uid, slices in grouped.items():
            logger.debug("Series %s: %d instances", uid, len(slices))
        return grouped

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------
    def sort_slices(self, slices: list[ImageData]) -> list[ImageData]:
        """Sort slices along the slice-normal direction.

        Uses the projection of ``ImagePositionPatient`` onto the slice
        normal (row x col), falling back to ``SliceLocation`` when all
        origins coincide.
        """
        if len(slices) <= 1:
            return list(slices)

        ref = slices[0]
        normal = self._slice_normal(ref)

        projections = [float(np.dot(np.asarray(s.origin), normal)) for s in slices]
        if max(projections) - min(projections) < 1e-6:
            # Degenerate (all origins identical) -> fall back to SliceLocation.
            def key(s: ImageData) -> float:
                return float(s.slice_location) if s.slice_location is not None else 0.0
            return sorted(slices, key=key)

        return [s for _, s in sorted(zip(projections, slices), key=lambda t: t[0])]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_series(self, slices: list[ImageData]) -> bool:
        """Check that slices form a consistent, stackable series.

        Verifies: same SeriesInstanceUID, consistent orientation,
        identical in-plane dimensions, consistent pixel spacing, and
        approximately uniform inter-slice spacing.
        """
        if not slices:
            return False
        if len(slices) == 1:
            return True

        ref = slices[0]
        ref_normal = self._slice_normal(ref)

        uids = {s.series_uid for s in slices}
        if len(uids) > 1:
            logger.warning("Series validation failed: %d distinct series UIDs", len(uids))
            return False

        for s in slices[1:]:
            # Orientation: normals must be parallel (same or opposite).
            if abs(float(np.dot(self._slice_normal(s), ref_normal))) < _ORIENTATION_TOLERANCE:
                logger.warning("Series validation failed: orientation mismatch")
                return False
            # Dimensions
            if s.array.shape[-2:] != ref.array.shape[-2:]:
                logger.warning("Series validation failed: dimension mismatch %s vs %s",
                               s.array.shape, ref.array.shape)
                return False
            # In-plane spacing
            if (abs(s.spacing[0] - ref.spacing[0]) > _SPACING_TOLERANCE
                    or abs(s.spacing[1] - ref.spacing[1]) > _SPACING_TOLERANCE):
                logger.warning("Series validation failed: spacing mismatch")
                return False

        # Inter-slice gap consistency (allow a few missing slices).
        ordered = self.sort_slices(slices)
        if len(ordered) >= 3:
            origins = np.array([s.origin for s in ordered], dtype=np.float64)
            gaps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
            gaps = gaps[gaps > 1e-6]
            if gaps.size >= 2 and (gaps.max() - gaps.min()) > 0.5 * gaps.mean():
                logger.warning("Series validation failed: non-uniform slice spacing")
                return False
        return True

    # ------------------------------------------------------------------
    # Volume construction
    # ------------------------------------------------------------------
    def build_volume(self, slices: list[ImageData], name: str = "Volume") -> VolumeData:
        """Sort and stack slices into a :class:`VolumeData`.

        Args:
            slices: Slices belonging to one series.
            name: Name for the resulting volume.

        Raises:
            ValueError: If the slice list is empty.
        """
        if not slices:
            raise ValueError("Cannot build a volume from an empty slice list")

        ordered = self.sort_slices(slices)
        volume = VolumeData.from_image_data_list(ordered, name=name)

        # Propagate modality/window info from the first slice.
        ref = ordered[0]
        volume.modality = getattr(ref, "modality", "") or volume.modality
        if ref.window_center is not None and ref.window_width is not None:
            volume.default_window = (float(ref.window_center), float(ref.window_width))
        volume.patient_name = getattr(ref, "patient_name", "")
        volume.patient_id = getattr(ref, "patient_id", "")
        volume.study_instance_uid = getattr(ref, "study_instance_uid", "")
        volume.series_instance_uid = ref.series_uid
        return volume

    # ------------------------------------------------------------------
    # Multi-frame support
    # ------------------------------------------------------------------
    def _expand_multiframe(self, images: list[ImageData]) -> list[ImageData]:
        """Expand multi-frame instances into per-frame ImageData objects."""
        result: list[ImageData] = []
        for image in images:
            num_frames = int(getattr(image, "num_frames", 1) or 1)
            if num_frames <= 1 or image.array.ndim != 3:
                result.append(image)
                continue

            per_frame_origins = self._per_frame_positions(image, num_frames)
            normal = self._slice_normal(image)
            for f in range(num_frames):
                origin = per_frame_origins[f] if per_frame_origins is not None else (
                    np.asarray(image.origin) + normal * f * getattr(image, "slice_thickness", 1.0)
                )
                frame = ImageData(
                    array=np.ascontiguousarray(image.array[f]),
                    spacing=image.spacing,
                    origin=tuple(float(v) for v in origin),  # type: ignore[arg-type]
                    orientation=image.orientation,
                    instance_uid=f"{image.instance_uid}.{f + 1}",
                    series_uid=image.series_uid,
                    sop_class_uid=image.sop_class_uid,
                    slice_location=None,
                    window_center=image.window_center,
                    window_width=image.window_width,
                    rescale_slope=image.rescale_slope,
                    rescale_intercept=image.rescale_intercept,
                )
                frame.num_frames = 1
                frame.instance_number = f + 1
                for attr in ("patient_name", "patient_id", "study_instance_uid",
                             "modality", "series_description", "slice_thickness"):
                    if hasattr(image, attr):
                        setattr(frame, attr, getattr(image, attr))
                result.append(frame)
            logger.info("Expanded multi-frame instance %s into %d frames",
                        image.instance_uid, num_frames)
        return result

    @staticmethod
    def _per_frame_positions(image: ImageData, num_frames: int) -> Optional[list[np.ndarray]]:
        """Try to read per-frame ImagePositionPatient (enhanced DICOM).

        Returns None when no per-frame information is available.
        """
        metadata = getattr(image, "metadata", None) or {}
        # DICOMReader does not currently carry the raw dataset; if a future
        # reader stores per-frame positions under metadata, honor them here.
        positions = metadata.get("PerFramePositions")
        if positions is not None:
            try:
                arr = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
                if arr.shape[0] == num_frames:
                    return [arr[i] for i in range(num_frames)]
            except Exception:  # noqa: BLE001
                pass
        return None

    @staticmethod
    def _slice_normal(image: ImageData) -> np.ndarray:
        """Unit slice normal from ImageOrientationPatient (row x col)."""
        row = np.asarray(image.orientation[:3], dtype=np.float64)
        col = np.asarray(image.orientation[3:6], dtype=np.float64)
        normal = np.cross(row, col)
        length = np.linalg.norm(normal)
        return normal / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
