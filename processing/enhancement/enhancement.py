"""
Enhancement — CLAHE / histogram equalization, window/level, log, gamma.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _clahe(volume, params, progress_callback=None):
    """Contrast-limited adaptive histogram equalization (scikit-image)."""
    import numpy as np
    from skimage.exposure import equalize_adapthist
    arr = np.asarray(volume.array, dtype=np.float64)
    kernel = int(params.get("kernel_size", 8))
    clip = float(params.get("clip_limit", 0.01))
    # equalize_adapthist expects images in [0, 1]; normalize per volume.
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / max(hi - lo, 1e-12)
    out = equalize_adapthist(norm, kernel_size=kernel, clip_limit=clip)
    return AlgorithmResult(algorithm_id="clahe", volume_data=out.astype(np.float32))


def _histogram_equalization(volume, params, progress_callback=None):
    """Global histogram equalization (scikit-image)."""
    import numpy as np
    from skimage.exposure import equalize_hist
    arr = np.asarray(volume.array, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / max(hi - lo, 1e-12)
    out = equalize_hist(norm)
    return AlgorithmResult(algorithm_id="histogram_equalization", volume_data=out.astype(np.float32))


def _window_level(volume, params, progress_callback=None):
    """Intensity windowing: clamp to [level - window/2, level + window/2]."""
    import numpy as np
    arr = np.asarray(volume.array, dtype=np.float64)
    window = float(params.get("window", 400))
    level = float(params.get("level", 40))
    lo = level - window / 2.0
    hi = level + window / 2.0
    out = np.clip(arr, lo, hi)
    return AlgorithmResult(algorithm_id="window_level", volume_data=out.astype(np.float32))


def _log_transform(volume, params, progress_callback=None):
    """Log intensity transform: log(1 + shift) * scale."""
    import numpy as np
    arr = np.asarray(volume.array, dtype=np.float64)
    shift = float(params.get("shift", 0.0))
    scale = float(params.get("scale", 1.0))
    out = np.log1p(np.maximum(arr + shift, 0.0)) * scale
    return AlgorithmResult(algorithm_id="log_transform", volume_data=out.astype(np.float32))


def _gamma(volume, params, progress_callback=None):
    """Gamma intensity correction."""
    import numpy as np
    arr = np.asarray(volume.array, dtype=np.float64)
    gamma = float(params.get("gamma", 1.0))
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / max(hi - lo, 1e-12)
    out = np.power(np.clip(norm, 0.0, 1.0), gamma)
    return AlgorithmResult(algorithm_id="gamma", volume_data=out.astype(np.float32))


def register_enhancement_algorithms():
    _registry.register(AlgorithmDefinition(
        id="clahe", name="CLAHE", category=AlgorithmCategory.ENHANCEMENT,
        description="Contrast-limited adaptive histogram equalization.",
        parameters=[
            AlgorithmParameter("kernel_size", "int", "Kernel Size", default=8, min_val=2, max_val=64),
            AlgorithmParameter("clip_limit", "float", "Clip Limit", default=0.01, min_val=0.001, max_val=0.5),
        ],
        run_func=_clahe,
    ))
    _registry.register(AlgorithmDefinition(
        id="histogram_equalization", name="Histogram Equalization", category=AlgorithmCategory.ENHANCEMENT,
        description="Global histogram equalization.",
        parameters=[], run_func=_histogram_equalization,
    ))
    _registry.register(AlgorithmDefinition(
        id="window_level", name="Window / Level", category=AlgorithmCategory.ENHANCEMENT,
        description="Intensity windowing (clamp to window around level).",
        parameters=[
            AlgorithmParameter("window", "float", "Window", default=400.0, min_val=1.0),
            AlgorithmParameter("level", "float", "Level", default=40.0),
        ],
        run_func=_window_level,
    ))
    _registry.register(AlgorithmDefinition(
        id="log_transform", name="Log Transform", category=AlgorithmCategory.ENHANCEMENT,
        description="log(1 + (value + shift)) * scale.",
        parameters=[
            AlgorithmParameter("shift", "float", "Shift", default=0.0),
            AlgorithmParameter("scale", "float", "Scale", default=1.0),
        ],
        run_func=_log_transform,
    ))
    _registry.register(AlgorithmDefinition(
        id="gamma", name="Gamma Correction", category=AlgorithmCategory.ENHANCEMENT,
        description="Normalized intensity gamma correction.",
        parameters=[AlgorithmParameter("gamma", "float", "Gamma", default=1.0, min_val=0.1, max_val=5.0)],
        run_func=_gamma,
    ))
