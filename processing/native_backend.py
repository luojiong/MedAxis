"""Native execution backend — every algorithm runs in C++/ITK when possible.

The Python layer only orchestrates: it converts a VolumeData to a numpy
buffer, dispatches to the compiled ``medaxis_itk`` module, and wraps the
result.  Algorithms without a native implementation fall back to their
Python ``run_func``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.native_extensions import get_native_module

# algorithm_id -> (medaxis_itk function name, native kind)
_FILTERING = {
    "gaussian": "gaussian",
    "median": "median",
    "bilateral": "bilateral",
    "anisotropic_diffusion": "anisotropic_diffusion",
    "nl_means": "nl_means",
    "gradient_magnitude": "gradient_magnitude",
    "laplacian": "laplacian",
    "sobel": "sobel",
    "canny": "canny",
    "zero_crossing": "zero_crossing",
    "mean": "mean",
    "discrete_gaussian": "discrete_gaussian",
    "normalize": "normalize",
    "rescale": "rescale",
    "cast": "cast",
    "sharpen": "sharpen",
}

_MORPHOLOGY = {
    "binary_erode": "binary_erode",
    "binary_dilate": "binary_dilate",
    "binary_open": "binary_open",
    "binary_close": "binary_close",
    "grayscale_erode": "grayscale_erode",
    "grayscale_dilate": "grayscale_dilate",
    "grayscale_open": "grayscale_open",
    "grayscale_close": "grayscale_close",
    "grayscale_tophat": "grayscale_tophat",
    "grayscale_bottomhat": "grayscale_bottomhat",
    "grayscale_gradient": "grayscale_gradient",
    "hole_filling": "hole_filling",
    "connected_components": "connected_components",
}

_THRESHOLD = {
    "otsu": "otsu",
    "adaptive": "adaptive",
    "isodata": "isodata",
    "max_entropy": "max_entropy",
    "moments": "moments",
    "renyi_entropy": "renyi_entropy",
    "huang": "huang",
    "intermodes": "intermodes",
    "li": "li",
    "manual_threshold": "manual",
}

_SEGMENTATION = {
    "connected_threshold": "connected_threshold",
    "confidence_connected": "confidence_connected",
    "neighborhood_connected": "neighborhood_connected",
    "watershed": "watershed",
    "fast_marching": "fast_marching",
    "geodesic_active_contour": "geodesic_active_contour",
    "shape_detection": "shape_detection",
    "threshold_segmentation": "threshold_segmentation",
    "laplacian_segmentation": "laplacian_segmentation",
    "kmeans": "kmeans",
    "fcm": "fcm",
    "island_remove": "island_remove",
}

_ARITHMETIC = {
    "arith_add": "add",
    "arith_subtract": "subtract",
    "arith_multiply": "multiply",
    "arith_divide": "divide",
    "mask": "mask",
    "invert": "invert",
}

_ENHANCEMENT = {
    "clahe": "clahe",
    "histogram_equalization": "histogram_equalization",
    "window_level": "window_level",
    "log_transform": "log_transform",
    "gamma": "gamma",
}

_GEOMETRY = {
    "flip": "flip",
    "permute": "permute",
    "crop": "crop",
    "pad": "pad",
    "resample": "resample",
}

_REGISTRATION = {
    "rigid": "rigid",
    "affine": "affine",
    "bspline": "bspline",
    "demons": "demons",
    "diffeomorphic_demons": "diffeomorphic_demons",
    "fast_symmetric_demons": "fast_symmetric_demons",
}

# Registration ids that need the second (moving) volume from params.
_REGISTRATION_IDS = set(_REGISTRATION)

# Threshold ids that use the native threshold_3d signature.
_THRESHOLD_IDS = set(_THRESHOLD)


def _to_float32(volume_or_array: Any) -> np.ndarray:
    arr = volume_or_array
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)


def run_native(algo_id: str, input_data: Any, params: dict) -> Optional[np.ndarray]:
    """Execute ``algo_id`` on the compiled backend.

    Returns the result array, or ``None`` when the algorithm has no native
    implementation (the Python fallback then runs).
    """
    itk = get_native_module("medaxis_itk")
    if itk is None:
        return None

    try:
        values = _to_float32(input_data)
    except Exception:
        return None

    if algo_id in _FILTERING:
        return np.asarray(itk.filter_3d(values, _FILTERING[algo_id], params))
    if algo_id in _MORPHOLOGY:
        return np.asarray(itk.morphology_3d(values, _MORPHOLOGY[algo_id], params))
    if algo_id in _THRESHOLD_IDS:
        native_kind = _THRESHOLD[algo_id]
        if native_kind == "manual":
            return np.asarray(itk.threshold_3d(
                values, method="manual",
                lower=float(params.get("lower", -100)),
                upper=float(params.get("upper", 300))))
        if native_kind == "adaptive":
            return np.asarray(itk.threshold_3d(
                values, method="adaptive",
                adaptive_radius=int(params.get("radius", 5))))
        return np.asarray(itk.threshold_3d(
            values, method=native_kind,
            histogram_bins=int(params.get("bins", 128))))
    if algo_id in _SEGMENTATION:
        return np.asarray(itk.segment_3d(values, _SEGMENTATION[algo_id], params))
    if algo_id in _ARITHMETIC:
        kind = _ARITHMETIC[algo_id]
        if kind == "mask":
            mask = params.get("mask")
            if mask is None:
                return None  # Python fallback raises a proper error
            return np.asarray(itk.arith_3d(
                values, _to_float32(mask), "mask", params))
        operand = params.get("operand")
        if operand is not None:
            b = _to_float32(operand)
        else:
            b = float(params.get("constant", 0.0))
        return np.asarray(itk.arith_3d(values, b, kind, params))
    if algo_id in _ENHANCEMENT:
        return np.asarray(itk.enhance_3d(values, _ENHANCEMENT[algo_id], params))
    if algo_id in _GEOMETRY:
        return np.asarray(itk.geometry_3d(values, _GEOMETRY[algo_id], params))
    if algo_id in _REGISTRATION_IDS:
        moving = params.get("moving")
        if moving is None:
            return None
        return np.asarray(itk.register_pair(
            values, _to_float32(moving), _REGISTRATION[algo_id], params))
    return None


def _safe(itk, algo_id, values, params):
    try:
        return run_native(algo_id, values, params)
    except Exception:
        return None
