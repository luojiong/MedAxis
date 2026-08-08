"""NIfTI writing — volumes and label maps via nibabel."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from core.volume_data import VolumeData
from core.label_data import LabelData

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:  # pragma: no cover
    nib = None  # type: ignore[assignment]
    HAS_NIBABEL = False

logger = logging.getLogger(__name__)

__all__ = ["NIfTIWriter"]

# NIfTI-1 intent codes
_NIFTI_INTENT_NONE = 0
_NIFTI_INTENT_LABEL = 1002   # NIFTI_INTENT_LABEL


class NIfTIWriter:
    """Write :class:`VolumeData` / :class:`LabelData` to ``.nii.gz`` files.

    MedAxis volumes are stored in LPS; nibabel expects RAS+ affines, so
    the affine is converted (first two axes negated) while the array is
    left untouched.
    """

    def write_volume(self, volume: VolumeData, path: Union[str, Path]) -> Path:
        """Write a volume as ``.nii.gz`` (float32)."""
        self._require_nibabel()
        path = self._ensure_nii_ext(Path(path))
        affine_ras = self._lps_affine_to_ras(volume.affine)

        data = volume.array.astype(np.float32)
        img = nib.Nifti1Image(data, affine_ras)
        hdr = img.header
        hdr.set_xyzt_units("mm", "sec")
        hdr.set_data_dtype(np.float32)
        hdr["descrip"] = f"MedAxis volume: {volume.name}"[:79].encode("ascii", errors="replace")

        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(path))
        logger.info("Wrote NIfTI volume: %s", path)
        return path

    def write_label(self, label_data: LabelData, path: Union[str, Path],
                    volume: VolumeData = None) -> Path:  # type: ignore[assignment]
        """Write a label map as ``.nii.gz`` with intent code NIFTI_INTENT_LABEL.

        Args:
            label_data: The label map to write (int16).
            path: Output path.
            volume: Parent volume providing geometry. When None, isotropic
                1 mm identity geometry is used.
        """
        self._require_nibabel()
        path = self._ensure_nii_ext(Path(path))

        if volume is not None:
            affine_ras = self._lps_affine_to_ras(volume.affine)
        else:
            affine_ras = np.eye(4, dtype=np.float64)

        data = label_data.array.astype(np.int16)
        img = nib.Nifti1Image(data, affine_ras)
        hdr = img.header
        hdr.set_data_dtype(np.int16)
        hdr.set_xyzt_units("mm", "sec")
        hdr["intent_code"] = _NIFTI_INTENT_LABEL
        hdr["intent_name"] = (label_data.name or "label")[:15].encode("ascii", errors="replace")
        hdr["descrip"] = f"MedAxis label: {label_data.name}"[:79].encode("ascii", errors="replace")

        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(path))
        logger.info("Wrote NIfTI label: %s", path)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_nibabel() -> None:
        if not HAS_NIBABEL:
            raise RuntimeError("nibabel is required for NIfTI writing")

    @staticmethod
    def _ensure_nii_ext(path: Path) -> Path:
        name = path.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            path = path.with_name(path.name + ".nii.gz")
        return path

    @staticmethod
    def _lps_affine_to_ras(affine_lps: np.ndarray) -> np.ndarray:
        """Convert an LPS voxel->world affine to RAS (nibabel convention)."""
        flip = np.eye(4, dtype=np.float64)
        flip[0, 0] = -1.0
        flip[1, 1] = -1.0
        return flip @ np.asarray(affine_lps, dtype=np.float64)
