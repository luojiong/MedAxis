"""Wavefront OBJ (+MTL) writing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from core.mesh_data import MeshData
from .base import BaseWriter

logger = logging.getLogger(__name__)

__all__ = ["OBJWriter"]


class OBJWriter(BaseWriter):
    """Write :class:`MeshData` as OBJ with an accompanying MTL material."""

    supported_extensions = (".obj",)

    def write(self, data: MeshData, path: Union[str, Path],
              material_name: str = "medaxis_default", **kwargs) -> Path:
        """Write the mesh as ``.obj`` plus a ``.mtl`` file.

        Args:
            data: The mesh to write.
            path: Destination ``.obj`` path.
            material_name: Name for the material entry.

        Returns:
            The written OBJ path.
        """
        mesh = data
        path = Path(path)
        if path.suffix.lower() != ".obj":
            path = path.with_suffix(".obj")
        path.parent.mkdir(parents=True, exist_ok=True)
        mtl_path = path.with_suffix(".mtl")

        color = self._display_color(mesh)
        self._write_mtl(mtl_path, material_name, color)
        self._write_obj(path, mesh, mtl_path.name, material_name)
        logger.info("Wrote OBJ: %s (+%s)", path, mtl_path.name)
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _display_color(mesh: MeshData) -> tuple[float, float, float]:
        """Derive a diffuse color from mesh data (default: light grey)."""
        if mesh.colors is not None and mesh.colors.shape[0] > 0:
            mean = np.asarray(mesh.colors[:, :3], dtype=np.float64).mean(axis=0) / 255.0
            return (float(mean[0]), float(mean[1]), float(mean[2]))
        return (0.75, 0.75, 0.78)

    @staticmethod
    def _write_obj(path: Path, mesh: MeshData, mtl_filename: str,
                   material_name: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# MedAxis OBJ export: {mesh.name}\n")
            f.write(f"mtllib {mtl_filename}\n")
            f.write(f"o {mesh.name.replace(' ', '_')}\n")

            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if mesh.normals is not None:
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write(f"usemtl {material_name}\n")
            f.write("s off\n")

            if mesh.normals is not None:
                for face in mesh.faces:
                    a, b, c = face + 1  # OBJ indices are 1-based
                    f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                for face in mesh.faces:
                    a, b, c = face + 1
                    f.write(f"f {a} {b} {c}\n")

    @staticmethod
    def _write_mtl(path: Path, material_name: str,
                   diffuse: tuple[float, float, float]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# MedAxis MTL export\n")
            f.write(f"newmtl {material_name}\n")
            f.write("Ka 0.100000 0.100000 0.100000\n")
            f.write(f"Kd {diffuse[0]:.6f} {diffuse[1]:.6f} {diffuse[2]:.6f}\n")
            f.write("Ks 0.200000 0.200000 0.200000\n")
            f.write("Ns 60.000000\n")
            f.write("d 1.000000\n")
            f.write("illum 2\n")
