"""NIfTI reading — .nii and .nii.gz via nibabel."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from core.volume_data import VolumeData

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:  # pragma: no cover
    nib = None  # type: ignore[assignment]
    HAS_NIBABEL = False

logger = logging.getLogger(__name__)

__all__ = ["NIfTIReader"]


class NIfTIReader:
    """Read NIfTI-1/2 files (.nii, .nii.gz) into :class:`VolumeData`.

    nibabel affines map voxel indices to RAS+ world coordinates. MedAxis
    volumes use the DICOM/ITK LPS convention, so the affine is converted
    (two leading axes negated) during import.
    """

    def read(self, path: Union[str, Path], dtype: type = np.float32) -> VolumeData:
        """Read a NIfTI file into a :class:`VolumeData`.

        Args:
            path: Path to a ``.nii`` or ``.nii.gz`` file.
            dtype: Numpy dtype for the loaded array (default float32).

        Returns:
            The loaded volume. Header metadata (intent, descrip, qform,
            sform) is attached as ``volume.metadata``.

        Raises:
            RuntimeError: If nibabel is unavailable.
            FileNotFoundError: If the file does not exist.
        """
        if not HAS_NIBABEL:
            raise RuntimeError("nibabel is required to read NIfTI files")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"NIfTI file not found: {path}")

        img = nib.load(str(path))
        array = np.asarray(img.get_fdata(caching="unchanged"), dtype=dtype)
        if array.ndim > 3:
            logger.warning(
                "NIfTI file %s has %d dimensions; keeping the first volume.",
                path.name, array.ndim,
            )
            array = array[..., 0] if array.ndim == 4 else array.reshape(array.shape[:3])

        affine_ras = np.asarray(img.affine, dtype=np.float64)
        spacing, origin, direction = self._affine_to_geometry(affine_ras)

        # RAS -> LPS conversion: negate the first two world axes.
        flip = np.diag([-1.0, -1.0, 1.0])
        origin_lps = tuple(float(v) for v in flip @ np.asarray(origin))
        dir_mat = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        direction_lps = tuple(float(v) for v in (flip @ dir_mat).reshape(-1))

        volume = VolumeData(
            array=array,
            spacing=spacing,
            origin=origin_lps,  # type: ignore[arg-type]
            direction=direction_lps,
            name=path.stem.replace(".nii", ""),
            modality=self._guess_modality(img.header),
        )
        volume.metadata = self._extract_metadata(img)
        volume.source_path = str(path)
        logger.info(
            "Loaded NIfTI %s: shape=%s spacing=%s", path.name, array.shape, spacing
        )
        return volume

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _affine_to_geometry(
        affine: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]:
        """Decompose a 4x4 affine into (spacing, origin, direction).

        Returns values in the affine's native (RAS) space; the caller is
        responsible for converting to LPS.
        """
        rotation_scaling = affine[:3, :3]
        spacing = np.linalg.norm(rotation_scaling, axis=0)
        spacing[spacing == 0.0] = 1.0
        direction = rotation_scaling / spacing[np.newaxis, :]
        origin = affine[:3, 3]
        return (
            tuple(float(s) for s in spacing),  # type: ignore[return-value]
            tuple(float(o) for o in origin),  # type: ignore[return-value]
            tuple(float(v) for v in direction.reshape(-1)),
        )

    @staticmethod
    def _extract_metadata(img) -> dict:
        """Extract header metadata: intent, descrip, qform/sform."""
        header = img.header
        meta: dict = {}
        try:
            meta["intent_code"] = int(header["intent_code"])
            meta["intent_name"] = str(header.get("intent_name", b"").tostring().decode(
                "ascii", errors="replace").strip("\x00"))
            meta["descrip"] = str(header.get("descrip", b"").tostring().decode(
                "ascii", errors="replace").strip("\x00"))
            meta["qform_code"] = int(header["qform_code"])
            meta["sform_code"] = int(header["sform_code"])
            qform = getattr(header, "get_qform", None)
            sform = getattr(header, "get_sform", None)
            if callable(qform):
                meta["qform"] = np.asarray(qform(), dtype=np.float64).tolist()
            if callable(sform):
                meta["sform"] = np.asarray(sform(), dtype=np.float64).tolist()
            meta["pixdim"] = [float(v) for v in np.asarray(header["pixdim"][1:4])]
            meta["datatype"] = int(header["datatype"])
            meta["xyzt_units"] = int(header["xyzt_units"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Partial NIfTI metadata extraction: %s", exc)
        return meta

    @staticmethod
    def _guess_modality(header) -> str:
        """Best-effort modality guess from the header description."""
        try:
            descrip = str(header.get("descrip", b"").tostring().decode(
                "ascii", errors="replace")).lower()
        except Exception:  # noqa: BLE001
            return ""
        for token, modality in (("ct", "CT"), ("mr", "MR"), ("mri", "MR"),
                                ("pet", "PT"), ("us", "US")):
            if token in descrip:
                return modality
        return ""
