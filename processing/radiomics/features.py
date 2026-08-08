"""First-order radiomics measurements with a compiled backend when present."""

from __future__ import annotations

import numpy as np

from core.native_extensions import get_native_module
from ..algorithm_registry import AlgorithmCategory, AlgorithmDefinition, AlgorithmRegistry, AlgorithmResult

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
