"""First-order radiomics measurements with a compiled backend when present."""

from __future__ import annotations

import numpy as np

from core.native_extensions import get_native_module
from ..algorithm_registry import AlgorithmCategory, AlgorithmDefinition, AlgorithmParameter, AlgorithmRegistry, AlgorithmResult

_registry = AlgorithmRegistry()


def _first_order(volume, params, progress_callback=None):
    values = np.asarray(volume.to_numpy(), dtype=np.float64)
    mask = params.get("mask")
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    else:
        values = values.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return AlgorithmResult(algorithm_id="first_order_radiomics", success=False, error="No finite voxels")

    native = get_native_module("medaxis_radiomics")
    if native is not None:
        stats = dict(native.first_order_features(values))
    else:
        stats = {
            "count": float(values.size),
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median": float(np.median(values)),
            "skewness": float(((values - values.mean()) ** 3).mean() / max(values.std() ** 3, 1.0e-12)),
            "kurtosis": float(((values - values.mean()) ** 4).mean() / max(values.std() ** 4, 1.0e-12) - 3.0),
        }
    return AlgorithmResult(algorithm_id="first_order_radiomics", statistics=stats)


def register_radiomics_algorithms() -> None:
    _registry.register(AlgorithmDefinition(
        id="first_order_radiomics",
        name="First-Order Radiomics",
        category=AlgorithmCategory.RADIOMICS,
        description="Intensity distribution statistics for the active volume or mask.",
        run_func=_first_order,
    ))


def _full_radiomics(volume, params, progress_callback=None):
    """119-feature suite: first-order + GLCM + GLRLM + GLSZM + GLDM + NGTDM + shape."""
    import numpy as np
    from .texture import extract_texture_features
    from .shape import extract_shape_features

    mask = params.get("mask")
    families = params.get("families", ["glcm", "glrlm", "glszm", "gldm", "ngtdm"])
    stats: dict = {}

    # First-order over the masked region (fall back to the whole volume).
    values = np.asarray(volume.to_numpy(), dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    else:
        values = values.ravel()
    values = values[np.isfinite(values)]
    if values.size:
        native = get_native_module("medaxis_radiomics")
        if native is not None:
            stats.update(dict(native.first_order_features(values)))
        else:
            stats["mean"] = float(values.mean())
            stats["variance"] = float(values.var())
            stats["standard_deviation"] = float(values.std())
            stats["minimum"] = float(values.min())
            stats["maximum"] = float(values.max())
            stats["median"] = float(np.median(values))

    # Texture features on the (masked) volume: prefer the compiled backend.
    arr = np.asarray(volume.to_numpy(), dtype=np.float32)
    if mask is not None:
        arr = np.where(np.asarray(mask, dtype=bool), arr, 0.0)
    levels = int(params.get("levels", 32))
    native = get_native_module("medaxis_radiomics")
    if native is not None:
        try:
            stats.update(dict(native.texture_features(
                np.ascontiguousarray(arr), levels, families)))
        except Exception as exc:  # pragma: no cover - defensive
            stats["texture_error"] = str(exc)[:200]
    else:
        try:
            stats.update(extract_texture_features(arr, families=families, levels=levels))
        except Exception as exc:  # pragma: no cover - defensive
            stats["texture_error"] = str(exc)[:200]

    # Shape features of the label mask.
    if mask is not None:
        spacing = getattr(volume, "spacing_mm", (1.0, 1.0, 1.0))
        mask_arr = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
        try:
            stats.update(extract_shape_features(
                (np.asarray(mask_arr) > 0).astype(np.uint8), spacing=spacing))
        except Exception as exc:  # pragma: no cover - defensive
            stats["shape_error"] = str(exc)[:200]

    return AlgorithmResult(algorithm_id="full_radiomics", statistics=stats)


def register_full_radiomics() -> None:
    _registry.register(AlgorithmDefinition(
        id="full_radiomics",
        name="Full Radiomics Suite (119 features)",
        category=AlgorithmCategory.RADIOMICS,
        description="First-order + GLCM(24) + GLRLM(16) + GLSZM(16) + GLDM(14) + "
                    "NGTDM(5) + Shape(2D/3D). Provide a label mask to include shape.",
        parameters=[
            AlgorithmParameter("mask", "volume", "Label Mask (optional)"),
            AlgorithmParameter("levels", "int", "Quantization Levels", default=32, min_val=8, max_val=256),
        ],
        run_func=_full_radiomics,
        estimated_time_sec=20.0,
    ))
