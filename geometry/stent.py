"""
Stent planning — circular (angular) unwrap of a vessel around its centerline.

For every centerline point the Frenet frame (T, N, B) defines a plane
perpendicular to the vessel; sampling along angles θ ∈ [0, 2π) at radius R
produces the classic "stent view" (angle × arc-length map), which shows the
vessel wall opened around the lumen.
"""
from __future__ import annotations


import numpy as np

from .centerline import centerline_to_bspline, extract_centerline


def circular_unwrap(
    volume_array: np.ndarray,
    centerline: np.ndarray,
    frames,
    radius_mm: float = 5.0,
    angular_samples: int = 360,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    radius_samples: int = 40,
) -> dict:
    """Unwrap a circular ring around the centerline into a 2-D map.

    Returns:
        dict with ``map`` (radius x angle x path? no — angle x path image),
        actually a (angular_samples, path_length) intensity map at the given
        radius band (average over radius_samples shells).
    """
    points = np.asarray(centerline, dtype=float)
    n_pts = len(points)
    if n_pts < 4 or frames is None or len(frames) < n_pts:
        raise ValueError("circular_unwrap needs >= 4 centerline points with frames")

    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    voxel = (points - origin) / spacing
    voxel = np.clip(voxel, 0, np.asarray(volume_array.shape) - 1)

    # Radius shells in mm.
    shells = np.linspace(0.5, radius_mm, radius_samples)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)

    map_out = np.full((angular_samples, n_pts), -np.inf, dtype=np.float32)
    for si, shell in enumerate(shells):
        for ai, theta in enumerate(angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for pi in range(n_pts):
                n = np.asarray(frames[pi].normal, dtype=float)
                b = np.asarray(frames[pi].binormal, dtype=float)
                offset_mm = shell * (cos_t * n + sin_t * b)
                sample_voxel = voxel[pi] + offset_mm / spacing
                # Max-intensity projection across radius shells (MIP).
                map_out[ai, pi] = max(map_out[ai, pi], _trilinear(volume_array, sample_voxel))
    map_out = np.where(np.isfinite(map_out), map_out, 0.0).astype(np.float32)

    return {
        "map": map_out,
        "angle_axis": angles,
        "path_points": points,
        "radius_mm": radius_mm,
    }


def _trilinear(volume: np.ndarray, voxel) -> float:
    k, j, i = voxel
    k0, j0, i0 = int(np.floor(k)), int(np.floor(j)), int(np.floor(i))
    dk, dj, di = k - k0, j - j0, i - i0
    shape = volume.shape
    if not (0 <= k0 < shape[0] - 1 and 0 <= j0 < shape[1] - 1 and 0 <= i0 < shape[2] - 1):
        return 0.0
    c000 = float(volume[k0, j0, i0])
    c100 = float(volume[k0 + 1, j0, i0])
    c010 = float(volume[k0, j0 + 1, i0])
    c110 = float(volume[k0 + 1, j0 + 1, i0])
    c001 = float(volume[k0, j0, i0 + 1])
    c101 = float(volume[k0 + 1, j0, i0 + 1])
    c011 = float(volume[k0, j0 + 1, i0 + 1])
    c111 = float(volume[k0 + 1, j0 + 1, i0 + 1])
    return (
        c000 * (1 - dk) * (1 - dj) * (1 - di)
        + c100 * dk * (1 - dj) * (1 - di)
        + c010 * (1 - dk) * dj * (1 - di)
        + c110 * dk * dj * (1 - di)
        + c001 * (1 - dk) * (1 - dj) * di
        + c101 * dk * (1 - dj) * di
        + c011 * (1 - dk) * dj * di
        + c111 * dk * dj * di
    )


def stent_view_from_label(
    label: np.ndarray,
    volume_array: np.ndarray,
    radius_mm: float = 5.0,
    angular_samples: int = 360,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    radius_samples: int = 40,
) -> dict:
    """Full pipeline: centerline extraction → frames → circular unwrap."""
    center = extract_centerline(label, spacing=spacing, origin=origin)
    curve = centerline_to_bspline(center["points"], 200)
    frames = curve["frames"]
    if frames is None:
        raise ValueError("stent_view_from_label: frame computation failed")
    return circular_unwrap(
        volume_array, curve["points"], frames,
        radius_mm=radius_mm, angular_samples=angular_samples,
        spacing=spacing, origin=origin, radius_samples=radius_samples,
    )
