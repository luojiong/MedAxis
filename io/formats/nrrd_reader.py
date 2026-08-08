"""NRRD reading — .nrrd/.nhdr via pynrrd (fallback: ITK)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from core.volume_data import VolumeData
from .base import BaseReader

logger = logging.getLogger(__name__)

__all__ = ["NRRDReader"]

try:
    import nrrd  # type: ignore

    HAS_NRRD = True
except ImportError:  # pragma: no cover
    nrrd = None  # type: ignore[assignment]
    HAS_NRRD = False

try:
    import itk  # type: ignore

    HAS_ITK = True
except ImportError:  # pragma: no cover
    itk = None  # type: ignore[assignment]
    HAS_ITK = False

# LPS flip -> MedAxis convention uses LPS already, but ITK/pynrrd may
# report RAS depending on the file's "space" field.
_RAS_FLIP = np.diag([-1.0, -1.0, 1.0])


class NRRDReader(BaseReader):
    """Read NRRD files into :class:`VolumeData`.

    Uses ``pynrrd`` when installed, otherwise falls back to ITK.
    """

    supported_extensions = (".nrrd", ".nhdr")

    def read(self, path: Union[str, Path], **kwargs) -> VolumeData:
        """Read a NRRD file into a :class:`VolumeData`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"NRRD file not found: {path}")

        if HAS_NRRD:
            return self._read_pynrrd(path)
        if HAS_ITK:
            return self._read_itk(path)
        raise RuntimeError("NRRD reading requires the 'pynrrd' or 'itk' package")

    # ------------------------------------------------------------------
    # pynrrd path
    # ------------------------------------------------------------------
    def _read_pynrrd(self, path: Path) -> VolumeData:
        data, header = nrrd.read(str(path))
        if data.ndim > 3:
            # Drop leading non-domain axes (e.g. vector components).
            kinds = header.get("kinds", [])
            domain_idx = [i for i, k in enumerate(kinds) if k == "domain"]
            if len(domain_idx) == 3:
                data = np.moveaxis(data, domain_idx, [0, 1, 2])
            data = data.reshape(data.shape[-3:])

        spacing, origin, direction = self._geometry_from_header(header, data.shape)
        volume = VolumeData(
            array=np.asarray(data),
            spacing=spacing,
            origin=origin,
            direction=direction,
            name=path.stem,
        )
        volume.metadata = {k: _jsonable(v) for k, v in header.items()}
        volume.source_path = str(path)
        return volume

    @staticmethod
    def _geometry_from_header(header: dict, shape: tuple) -> tuple:
        """Extract (spacing, origin, direction) from a pynrrd header."""
        space = str(header.get("space", "right-anterior-superior")).lower()
        is_ras = space in ("right-anterior-superior", "ras")

        origin = np.zeros(3, dtype=np.float64)
        directions = np.eye(3, dtype=np.float64)
        spacing = np.ones(3, dtype=np.float64)

        if "space origin" in header:
            origin = np.asarray(header["space origin"], dtype=np.float64).reshape(3)
        if "space directions" in header:
            sd = np.asarray(header["space directions"], dtype=np.float64)
            # Rows may include non-domain axes marked NaN; keep 3x3 part.
            sd = sd[-3:] if sd.shape[0] > 3 else sd
            for axis in range(3):
                vec = sd[axis][:3]
                length = np.linalg.norm(vec)
                if np.isnan(length) or length < 1e-12:
                    continue
                spacing[axis] = length
                directions[:, axis] = vec / length

        if is_ras:
            origin = _RAS_FLIP @ origin
            directions = _RAS_FLIP @ directions

        return (
            tuple(float(s) for s in spacing),  # type: ignore[return-value]
            tuple(float(o) for o in origin),  # type: ignore[return-value]
            tuple(float(v) for v in directions.reshape(-1)),
        )

    # ------------------------------------------------------------------
    # ITK path
    # ------------------------------------------------------------------
    def _read_itk(self, path: Path) -> VolumeData:
        image = itk.imread(str(path))
        array = itk.GetArrayFromImage(image)  # (k, j, i)
        spacing = tuple(float(s) for s in image.GetSpacing())  # type: ignore[assignment]
        origin_lps = tuple(float(o) for o in image.GetOrigin())  # type: ignore[assignment]
        direction = tuple(float(v) for v in np.asarray(
            image.GetDirection(), dtype=np.float64).reshape(-1))
        volume = VolumeData(
            array=array,
            spacing=spacing,  # type: ignore[arg-type]
            origin=origin_lps,  # type: ignore[arg-type]
            direction=direction,
            name=path.stem,
        )
        volume.source_path = str(path)
        return volume


def _jsonable(value):
    """Best-effort conversion of header values for JSON storage."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    return value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)
