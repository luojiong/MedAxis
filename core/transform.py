"""DICOM coordinate transformation utilities.

DICOM / ITK use the LPS (Left, Posterior, Superior) patient coordinate
system. Voxel indices (i, j, k) map to world coordinates (mm) through an
affine matrix built from spacing, origin and direction cosines:

    world = direction @ diag(spacing) @ voxel + origin
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "build_affine_matrix",
    "world_to_voxel_coords",
    "voxel_to_world_coords",
    "LPS_to_RAS",
    "RAS_to_LPS",
    "lps_to_ras_matrix",
]


def _as_direction_matrix(direction) -> np.ndarray:
    """Normalize a direction specification to a 3x3 float array.

    Accepts a flat sequence of 9 values (ITK / DICOM
    ImageOrientationPatient-style row-major), a 3x3 array-like, or None
    (identity).
    """
    if direction is None:
        return np.eye(3, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if d.shape == (9,):
        return d.reshape(3, 3)
    if d.shape == (6,):
        # ImageOrientationPatient: row (first 3) and column (last 3)
        # direction cosines. Slice normal is their cross product.
        row = d[:3]
        col = d[3:]
        slice_normal = np.cross(row, col)
        return np.column_stack([row, col, slice_normal])
    if d.shape == (3, 3):
        return d
    raise ValueError(f"Cannot interpret direction with shape {d.shape}")


def build_affine_matrix(
    spacing,
    origin,
    direction=None,
) -> np.ndarray:
    """Build a 4x4 voxel->world affine matrix.

    Args:
        spacing: Voxel sizes (sx, sy, sz) in mm.
        origin: World coordinate (mm) of voxel (0, 0, 0).
        direction: 9-element flat direction cosines, 3x3 matrix, or
            6-element ImageOrientationPatient. Defaults to identity.

    Returns:
        4x4 ``numpy.ndarray`` mapping (i, j, k, 1) -> (x, y, z, 1).
    """
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    dir_mat = _as_direction_matrix(direction)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = dir_mat @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def world_to_voxel_coords(world_xyz, affine: np.ndarray) -> np.ndarray:
    """Convert world coordinates (mm) to (continuous) voxel indices.

    Args:
        world_xyz: Array of shape (3,) or (N, 3).
        affine: 4x4 voxel->world affine from :func:`build_affine_matrix`.

    Returns:
        Voxel coordinates (i, j, k) with the same leading shape as input.
    """
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    single = world_xyz.ndim == 1
    pts = world_xyz.reshape(-1, 3)

    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    homog = np.column_stack([pts, np.ones(len(pts))])
    ijk = (inv @ homog.T).T[:, :3]
    return ijk[0] if single else ijk


def voxel_to_world_coords(voxel_ijk, affine: np.ndarray) -> np.ndarray:
    """Convert voxel indices to world coordinates (mm).

    Args:
        voxel_ijk: Array of shape (3,) or (N, 3).
        affine: 4x4 voxel->world affine from :func:`build_affine_matrix`.

    Returns:
        World coordinates (x, y, z) with the same leading shape as input.
    """
    voxel_ijk = np.asarray(voxel_ijk, dtype=np.float64)
    single = voxel_ijk.ndim == 1
    pts = voxel_ijk.reshape(-1, 3)

    affine = np.asarray(affine, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts))])
    xyz = (affine @ homog.T).T[:, :3]
    return xyz[0] if single else xyz


def LPS_to_RAS(coords) -> np.ndarray:
    """Convert LPS coordinates to RAS (negate L and P axes)."""
    coords = np.asarray(coords, dtype=np.float64)
    sign = np.array([-1.0, -1.0, 1.0])
    return coords * sign


def RAS_to_LPS(coords) -> np.ndarray:
    """Convert RAS coordinates to LPS (negate R and A axes)."""
    # The transformation is symmetric / self-inverse.
    return LPS_to_RAS(coords)


def lps_to_ras_matrix() -> np.ndarray:
    """Return the 4x4 matrix converting LPS coordinates to RAS."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = -1.0
    m[1, 1] = -1.0
    return m
