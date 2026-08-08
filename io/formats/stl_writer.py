"""Binary STL writing."""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Union

import numpy as np

from core.mesh_data import MeshData
from .base import BaseWriter

logger = logging.getLogger(__name__)

__all__ = ["STLWriter"]

_HEADER_SIZE = 80


class STLWriter(BaseWriter):
    """Write :class:`MeshData` as binary STL."""

    supported_extensions = (".stl",)

    def write(self, data: MeshData, path: Union[str, Path], **kwargs) -> Path:
        """Write the mesh as a binary STL file.

        Args:
            data: The mesh to write.
            path: Destination path (``.stl`` appended if missing).

        Returns:
            The written path.
        """
        mesh = data
        path = Path(path)
        if path.suffix.lower() != ".stl":
            path = path.with_suffix(".stl")
        path.parent.mkdir(parents=True, exist_ok=True)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        normals = mesh._face_normals().astype(np.float32)

        header = f"MedAxis binary STL: {mesh.name}".encode("ascii", errors="replace")
        header = header[:_HEADER_SIZE].ljust(_HEADER_SIZE, b"\0")

        with open(path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", faces.shape[0]))

            triangles = vertices[faces]  # (M, 3, 3)
            for tri, normal in zip(triangles, normals):
                f.write(struct.pack("<3f", *normal))
                for vertex in tri:
                    f.write(struct.pack("<3f", *vertex))
                f.write(struct.pack("<H", 0))  # attribute byte count

        logger.info("Wrote binary STL: %s (%d triangles)", path, faces.shape[0])
        return path
