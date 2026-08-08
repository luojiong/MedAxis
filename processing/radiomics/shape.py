"""
Shape radiomics — 2D (10) and 3D (16) shape features of a label volume.
"""
from __future__ import annotations

import numpy as np


def _safe_div(a, b):
    return float(a / b) if b not in (0, 0.0) and not np.isnan(b) else 0.0


def shape_features_2d(label: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Per-slice 2D shape features, averaged over non-empty slices."""
    from skimage.measure import regionprops
    from skimage.measure import label as sklabel

    si, sj, sk = spacing
    all_feats = []
    for slice_idx in range(label.shape[0]):
        sl = label[slice_idx] > 0
        if not sl.any():
            continue
        lab = sklabel(sl)
        props = regionprops(lab, spacing=(sj, sk))
        for p in props:
            all_feats.append(p)
    if not all_feats:
        return {}

    def avg(name, fn):
        vals = [fn(p) for p in all_feats]
        return float(np.mean(vals))

    perimeters = [p.perimeter for p in all_feats]
    areas = [p.area for p in all_feats]
    total_perimeter = float(np.sum(perimeters))
    total_area = float(np.sum(areas))

    major = [p.axis_major_length for p in all_feats]
    minor = [p.axis_minor_length for p in all_feats]

    return {
        "shape2d_perimeter": total_perimeter,
        "shape2d_area": total_area,
        "shape2d_perimeter_surface_ratio": _safe_div(total_perimeter, total_area),
        "shape2d_sphericity": float(np.mean([_safe_div(2 * np.sqrt(np.pi * p.area), p.perimeter) for p in all_feats])),
        "shape2d_major_axis": float(np.mean(major)),
        "shape2d_minor_axis": float(np.mean(minor)),
        "shape2d_elongation": float(np.mean([_safe_div(mi, ma) for mi, ma in zip(minor, major)])),
        "shape2d_compactness": float(np.mean([_safe_div(4 * np.pi * p.area, p.perimeter**2) for p in all_feats])),
        "shape2d_max_diameter": float(np.max([p.axis_major_length for p in all_feats])),
        "shape2d_slice_count": len(all_feats),
    }


def shape_features_3d(label: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """3D shape features of the largest connected component."""
    from skimage.measure import regionprops
    from skimage.measure import label as sklabel

    lab = sklabel(label > 0)
    props = regionprops(lab, spacing=spacing)
    if not props:
        return {}
    p = max(props, key=lambda x: x.area)

    volume = p.area  # voxels
    volume_mm3 = p.area * spacing[0] * spacing[1] * spacing[2]
    surf = p.surface_area if hasattr(p, "surface_area") else 0.0
    # Principal axes from the inertia tensor (scaled to physical units).
    eigvals = sorted(p.inertia_tensor_eigvals)
    _major_axis, middle, minor = (
        float(eigvals[2]) * spacing[0],
        float(eigvals[1]) * spacing[1],
        float(eigvals[0]) * spacing[2],
    )
    max_diam = float(p.axis_major_length)

    # Bounding-box-based compactness/flatness (pyradiomics style).
    bbox = p.bbox  # (k0, j0, i0, k1, j1, i1)
    dk = (bbox[3] - bbox[0]) * spacing[0]
    dj = (bbox[4] - bbox[1]) * spacing[1]
    di = (bbox[5] - bbox[2]) * spacing[2]

    def sphericity():
        r = _safe_div(3 * volume_mm3, 4 * np.pi)
        r = r ** (1 / 3) if r > 0 else 0.0
        return _safe_div(4 * np.pi * r**2, surf) if surf > 0 else 0.0

    def compactness():
        return _safe_div(volume_mm3, np.sqrt(np.pi) * surf**1.5) if surf > 0 else 0.0

    bbox_major, bbox_middle, bbox_minor = sorted([di, dj, dk], reverse=True)

    return {
        "shape3d_volume_voxels": int(volume),
        "shape3d_volume_mm3": volume_mm3,
        "shape3d_surface_area_mm2": surf,
        "shape3d_surface_volume_ratio": _safe_div(surf, volume_mm3),
        "shape3d_sphericity": sphericity(),
        "shape3d_compactness": compactness(),
        "shape3d_flatness": _safe_div(minor, bbox_major),
        "shape3d_elongation": _safe_div(middle, bbox_major),
        "shape3d_max_3d_diameter": max_diam,
        "shape3d_major_axis": float(bbox_major),
        "shape3d_middle_axis": float(bbox_middle),
        "shape3d_minor_axis": float(bbox_minor),
        "shape3d_bbox_volume_mm3": float(di * dj * dk),
        "shape3d_voxel_volume": float(volume_mm3 / max(volume, 1)),
        "shape3d_component_count": len(props),
        "shape3d_lesion_volume_ratio": _safe_div(volume_mm3, di * dj * dk),
    }


def extract_shape_features(label: np.ndarray, spacing=(1.0, 1.0, 1.0), dims=("2d", "3d")) -> dict:
    out: dict = {}
    if "2d" in dims:
        out.update(shape_features_2d(label, spacing))
    if "3d" in dims:
        out.update(shape_features_3d(label, spacing))
    return out
