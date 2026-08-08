"""MeshData — triangular surface mesh data model."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

__all__ = ["MeshData"]


@dataclass
class MeshData:
    """A triangular surface mesh.

    Attributes:
        vertices: (N, 3) float array of vertex positions in mm.
        faces: (M, 3) int array of triangle vertex indices.
        normals: Optional (N, 3) per-vertex normals.
        colors: Optional per-vertex colors, (N, 3) or (N, 4), 0-255.
        name: Display name.
        visible: Whether the mesh is shown in the 3D view.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    name: str = "Mesh"
    visible: bool = True
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32).reshape(-1, 3)
            if self.normals.shape != self.vertices.shape:
                raise ValueError("normals must have the same shape as vertices")
        if self.colors is not None:
            self.colors = np.asarray(self.colors)
            if self.colors.shape[0] != self.vertex_count:
                raise ValueError("colors must have one entry per vertex")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Axis-aligned bounds (xmin, xmax, ymin, ymax, zmin, zmax) in mm."""
        if self.vertex_count == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mn = self.vertices.min(axis=0)
        mx = self.vertices.max(axis=0)
        return (float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2]))

    @property
    def surface_area(self) -> float:
        """Total surface area in mm^2."""
        if self.face_count == 0:
            return 0.0
        tri = self.vertices[self.faces]  # (M, 3, 3)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    @property
    def volume_ml(self) -> float:
        """Enclosed (signed) volume in mL, assuming a closed manifold mesh.

        Uses the divergence theorem. Counter-clockwise outward-facing
        winding gives a positive value.
        """
        if self.face_count == 0:
            return 0.0
        tri = self.vertices[self.faces].astype(np.float64)
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        vol_mm3 = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
        return float(abs(vol_mm3) / 1000.0)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_stl(self, path: Union[str, Path]) -> None:
        """Write the mesh as an ASCII STL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normals = self._face_normals() if self.normals is None else self.normals[self.faces].mean(axis=1)
        with open(path, "w", encoding="ascii") as f:
            f.write(f"solid {self.name}\n")
            for face, n in zip(self.faces, normals):
                f.write(f"facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("  outer loop\n")
                for vi in face:
                    v = self.vertices[vi]
                    f.write(f"    vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("  endloop\nendfacet\n")
            f.write(f"endsolid {self.name}\n")

    def save_obj(self, path: Union[str, Path]) -> None:
        """Write the mesh as a Wavefront OBJ file (1-based face indices)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# MedAxis mesh: {self.name}\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            if self.normals is not None:
                for n in self.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            if self.normals is not None:
                for face in self.faces:
                    a, b, c = face + 1
                    f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                for face in self.faces:
                    a, b, c = face + 1
                    f.write(f"f {a} {b} {c}\n")

    def save_ply(self, path: Union[str, Path]) -> None:
        """Write the mesh as a binary little-endian PLY file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_colors = self.colors is not None and self.colors.shape[1] >= 3

        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"comment MedAxis mesh: {self.name}",
            f"element vertex {self.vertex_count}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_colors:
            header_lines += [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        header_lines += [
            f"element face {self.face_count}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        header = "\n".join(header_lines) + "\n"

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(self.vertex_count):
                f.write(self.vertices[i].astype("<f4").tobytes())
                if has_colors:
                    f.write(np.clip(self.colors[i, :3], 0, 255).astype("u1").tobytes())
            for face in self.faces:
                f.write(np.array([3], dtype="u1").tobytes())
                f.write(face.astype("<i4").tobytes())

    # ------------------------------------------------------------------
    # Helpers / conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize mesh metadata; geometry is stored by ProjectIO."""
        return {
            "id": self.id,
            "name": self.name,
            "visible": self.visible,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
        }

    def _face_normals(self) -> np.ndarray:
        """Compute un-normalized face normals (M, 3)."""
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return n / norm

    def compute_vertex_normals(self) -> np.ndarray:
        """Compute area-weighted per-vertex normals and store them."""
        fn = self._face_normals()
        vn = np.zeros_like(self.vertices, dtype=np.float64)
        np.add.at(vn, self.faces[:, 0], fn)
        np.add.at(vn, self.faces[:, 1], fn)
        np.add.at(vn, self.faces[:, 2], fn)
        norm = np.linalg.norm(vn, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.normals = (vn / norm).astype(np.float32)
        return self.normals

    @classmethod
    def from_vtk_polydata(cls, vtk_poly, name: str = "Mesh") -> "MeshData":
        """Build a :class:`MeshData` from a ``vtk.vtkPolyData`` object."""
        points = np.asarray(vtk_poly.GetPoints().GetData(), dtype=np.float32)

        polys = vtk_poly.GetPolys()
        conn = np.asarray(polys.GetConnectivityArray(), dtype=np.int64)
        if conn.size % 3 != 0:
            raise ValueError("PolyData contains non-triangular cells; triangulate first")
        faces = conn.reshape(-1, 3)

        normals = None
        if vtk_poly.GetPointData().GetNormals() is not None:
            normals = np.asarray(vtk_poly.GetPointData().GetNormals(), dtype=np.float32)

        colors = None
        scalars = vtk_poly.GetPointData().GetScalars()
        if scalars is not None and scalars.GetNumberOfComponents() in (3, 4):
            colors = np.asarray(scalars)

        return cls(vertices=points, faces=faces, normals=normals, colors=colors, name=name)
