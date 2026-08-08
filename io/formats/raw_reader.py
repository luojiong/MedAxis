"""RAW volume reading — headerless binary voxel dumps."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from core.volume_data import VolumeData
from .base import BaseReader

logger = logging.getLogger(__name__)

__all__ = ["RAWReader", "RAW_DTYPES"]

#: Common raw dtype shorthands.
RAW_DTYPES = {
    "uint8": np.uint8, "uchar": np.uint8, "u8": np.uint8,
    "int16": np.int16, "short": np.int16, "i16": np.int16,
    "uint16": np.uint16, "ushort": np.uint16, "u16": np.uint16,
    "int32": np.int32, "int": np.int32, "i32": np.int32,
    "float32": np.float32, "float": np.float32, "f32": np.float32,
    "float64": np.float64, "double": np.float64, "f64": np.float64,
}


class RAWReader(BaseReader):
    """Read headerless RAW binary volumes.

    Since RAW files carry no metadata, dimensions, spacing and dtype must
    be supplied by the caller.
    """

    supported_extensions = (".raw", ".r16", ".r32", ".f32", ".u8", ".u16")

    def read(
        self,
        path: Union[str, Path],
        dims: Optional[tuple[int, int, int]] = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        dtype: Union[str, type] = "int16",
        offset: int = 0,
        little_endian: bool = True,
        **kwargs,
    ) -> VolumeData:
        """Read a RAW volume.

        Args:
            path: Binary file path.
            dims: Volume dimensions ``(nx, ny, nz)``. Required.
            spacing: Voxel size (sx, sy, sz) in mm.
            dtype: Numpy dtype or shorthand from :data:`RAW_DTYPES`.
            offset: Byte offset of the voxel data in the file.
            little_endian: Byte order of the file.

        Returns:
            The loaded :class:`VolumeData`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"RAW file not found: {path}")
        if dims is None:
            raise ValueError("RAW files have no header: 'dims' (nx, ny, nz) is required")
        nx, ny, nz = (int(v) for v in dims)

        np_dtype = RAW_DTYPES[str(dtype)] if isinstance(dtype, str) else np.dtype(dtype).type
        if np_dtype not in RAW_DTYPES.values():
            np_dtype = np.dtype(dtype).type

        byte_order = "<" if little_endian else ">"
        full_dtype = np.dtype(np_dtype).newbyteorder(byte_order)

        expected = nx * ny * nz * full_dtype.itemsize
        file_size = path.stat().st_size - offset
        if file_size < expected:
            raise ValueError(
                f"RAW file too small: need {expected} bytes for dims {dims} "
                f"({np.dtype(np_dtype).name}), file has {file_size}"
            )

        with open(path, "rb") as f:
            f.seek(offset)
            buffer = f.read(expected)
        array = np.frombuffer(buffer, dtype=full_dtype).reshape(nz, ny, nx)
        array = np.ascontiguousarray(array.astype(np_dtype))

        volume = VolumeData(
            array=array,
            spacing=tuple(float(s) for s in spacing),  # type: ignore[arg-type]
            origin=(0.0, 0.0, 0.0),
            name=path.stem,
        )
        volume.source_path = str(path)
        logger.info("Loaded RAW volume %s: dims=%s dtype=%s", path.name, dims, np_dtype.__name__)
        return volume
