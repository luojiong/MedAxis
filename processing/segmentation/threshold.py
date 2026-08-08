"""
Segmentation algorithms — thresholding, region growing, watershed, level set.
"""
from ..algorithm_registry import AlgorithmRegistry, AlgorithmDefinition, AlgorithmCategory, AlgorithmResult, AlgorithmParameter

_registry = AlgorithmRegistry()


def _otsu_threshold(volume, params, progress_callback=None):
    """Otsu multi-threshold segmentation."""
    from core.native_extensions import get_native_module
    import numpy as np
    num_thresholds = params.get("num_thresholds", 2)
    native = get_native_module("medaxis_itk")
    if native is not None and int(num_thresholds) == 1:
        result = native.threshold_3d(volume.to_numpy().astype(np.float32), method="otsu")
        return AlgorithmResult(algorithm_id="otsu", label_data=result)
    import itk
    itk_image = volume.to_itk_image()
    result = itk.otsu_multiple_thresholds_image_filter(itk_image, number_of_thresholds=num_thresholds)
    return AlgorithmResult(algorithm_id="otsu", label_data=itk.array_from_image(result))


def _manual_threshold(volume, params, progress_callback=None):
    """Manual range threshold."""
    import numpy as np
    from core.native_extensions import get_native_module
    lower = params.get("lower", -100)
    upper = params.get("upper", 300)
    native = get_native_module("medaxis_itk")
    if native is not None:
        result = native.threshold_3d(
            volume.to_numpy().astype(np.float32),
            method="manual",
            lower=float(lower),
            upper=float(upper),
        )
        return AlgorithmResult(algorithm_id="manual_threshold", label_data=result)
    arr = volume.to_numpy()
    label = np.where((arr >= lower) & (arr <= upper), 1, 0).astype(np.uint8)
    return AlgorithmResult(algorithm_id="manual_threshold", label_data=label)


def _region_growing(volume, params, progress_callback=None):
    """Connected threshold region growing."""
    import itk
    seed = params.get("seed", [0, 0, 0])
    lower = params.get("lower", 0)
    upper = params.get("upper", 255)
    itk_image = volume.to_itk_image()
    result = itk.connected_threshold_image_filter(
        itk_image,
        seed=seed,
        lower=lower,
        upper=upper,
        replace_value=1,
    )
    return AlgorithmResult(algorithm_id="region_growing", label_data=itk.array_from_image(result))


def _watershed(volume, params, progress_callback=None):
    """Watershed segmentation."""
    import itk
    level = params.get("level", 0.3)
    threshold = params.get("threshold", 0.001)
    itk_image = volume.to_itk_image()
    result = itk.watershed_image_filter(itk_image, level=level, threshold=threshold)
    return AlgorithmResult(algorithm_id="watershed", label_data=itk.array_from_image(result))


def _flood_fill(volume, params, progress_callback=None):
    """3D flood fill from seed point."""
    import numpy as np
    from scipy import ndimage
    seed = params.get("seed", [0, 0, 0])
    lower = params.get("lower", 0)
    upper = params.get("upper", 255)
    connectivity = params.get("connectivity", 26)

    arr = volume.to_numpy()
    mask = (arr >= lower) & (arr <= upper)
    structure = ndimage.generate_binary_structure(3, 3 if connectivity == 26 else 2 if connectivity == 18 else 1)
    seed_idx = tuple(seed)
    if not mask[seed_idx]:
        return AlgorithmResult(algorithm_id="flood_fill", success=False, error="Seed point outside threshold range")
    label = np.zeros_like(arr, dtype=np.uint8)
    label[seed_idx] = 1
    filled = ndimage.binary_dilation(label, structure=structure, mask=mask, iterations=100000)
    return AlgorithmResult(algorithm_id="flood_fill", label_data=filled.astype(np.uint8))


def register_segmentation_algorithms():
    _registry.register(AlgorithmDefinition(
        id="otsu", name="Otsu Threshold", category=AlgorithmCategory.SEGMENTATION,
        parameters=[AlgorithmParameter("num_thresholds", "int", "Number of Thresholds", default=2, min_val=1, max_val=10)],
        run_func=_otsu_threshold,
    ))
    _registry.register(AlgorithmDefinition(
        id="manual_threshold", name="Manual Threshold", category=AlgorithmCategory.SEGMENTATION,
        parameters=[
            AlgorithmParameter("lower", "float", "Lower Threshold", default=-100),
            AlgorithmParameter("upper", "float", "Upper Threshold", default=300),
        ],
        run_func=_manual_threshold,
    ))
    _registry.register(AlgorithmDefinition(
        id="region_growing", name="Region Growing", category=AlgorithmCategory.SEGMENTATION,
        parameters=[
            AlgorithmParameter("seed", "point3d", "Seed Point (i,j,k)", default=[0, 0, 0]),
            AlgorithmParameter("lower", "float", "Lower Threshold", default=0),
            AlgorithmParameter("upper", "float", "Upper Threshold", default=255),
        ],
        run_func=_region_growing,
    ))
    _registry.register(AlgorithmDefinition(
        id="watershed", name="Watershed", category=AlgorithmCategory.SEGMENTATION,
        parameters=[
            AlgorithmParameter("level", "float", "Level", default=0.3, min_val=0.0, max_val=1.0),
            AlgorithmParameter("threshold", "float", "Threshold", default=0.001, min_val=0.0, max_val=1.0),
        ],
        run_func=_watershed,
    ))
    _registry.register(AlgorithmDefinition(
        id="flood_fill", name="Flood Fill (泛洪填充)", category=AlgorithmCategory.SEGMENTATION,
        parameters=[
            AlgorithmParameter("seed", "point3d", "Seed Point (i,j,k)"),
            AlgorithmParameter("lower", "float", "Lower Threshold", default=0),
            AlgorithmParameter("upper", "float", "Upper Threshold", default=255),
            AlgorithmParameter("connectivity", "choice", "Connectivity", choices=["6", "18", "26"], default="26"),
        ],
        run_func=_flood_fill,
    ))


def _histogram_threshold(method: str, label: str):
    """Factory for single-histogram threshold methods (native-backed)."""

    def run(volume, params, progress_callback=None):
        from core.native_extensions import get_native_module
        import numpy as np
        native = get_native_module("medaxis_itk")
        bins = int(params.get("bins", 128))
        if native is not None:
            result = native.threshold_3d(
                volume.to_numpy().astype(np.float32),
                method=method,
                histogram_bins=bins,
            )
            return AlgorithmResult(algorithm_id=method, label_data=result)
        # Python fallback: ITK filter of the same family.
        import itk
        filter_name = {
            "adaptive": None,
            "isodata": "isodata_threshold_image_filter",
            "max_entropy": "maximum_entropy_threshold_image_filter",
            "moments": "moments_threshold_image_filter",
            "renyi_entropy": "renyi_entropy_threshold_image_filter",
            "huang": "huang_threshold_image_filter",
            "intermodes": "intermodes_threshold_image_filter",
            "li": "li_threshold_image_filter",
        }[method]
        if filter_name is None:
            raise RuntimeError("adaptive threshold requires the native medaxis_itk module")
        itk_image = volume.to_itk_image()
        result = getattr(itk, filter_name)(itk_image, number_of_histogram_bins=bins)
        return AlgorithmResult(algorithm_id=method, label_data=itk.array_from_image(result))

    run.__name__ = f"_{method}"
    return run


def _adaptive_threshold(volume, params, progress_callback=None):
    """Local-mean adaptive threshold (Bradley)."""
    from core.native_extensions import get_native_module
    import numpy as np
    native = get_native_module("medaxis_itk")
    radius = int(params.get("radius", 5))
    if native is not None:
        result = native.threshold_3d(
            volume.to_numpy().astype(np.float32),
            method="adaptive",
            adaptive_radius=radius,
        )
        return AlgorithmResult(algorithm_id="adaptive", label_data=result)
    raise RuntimeError("adaptive threshold requires the native medaxis_itk module")


_THRESHOLD_METHODS = [
    ("adaptive", "Adaptive Threshold", "Local-mean (Bradley) adaptive thresholding.",
     [AlgorithmParameter("radius", "int", "Local Radius", default=5, min_val=1, max_val=20)]),
    ("isodata", "IsoData Threshold", "IsoData automatic threshold (Ridler-Calvard).",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("max_entropy", "Maximum Entropy", "Maximum entropy threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("moments", "Moments Threshold", "Moment-preserving threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("renyi_entropy", "Renyi Entropy", "Renyi entropy threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("huang", "Huang Threshold", "Huang fuzzy threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("intermodes", "Intermodes Threshold", "Intermodes (histogram valley) threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
    ("li", "Li Threshold", "Li minimum cross-entropy threshold.",
     [AlgorithmParameter("bins", "int", "Histogram Bins", default=128, min_val=16, max_val=1024)]),
]


def register_histogram_thresholds():
    for method, name, desc, params in _THRESHOLD_METHODS:
        run = _adaptive_threshold if method == "adaptive" else _histogram_threshold(method, name)
        _registry.register(AlgorithmDefinition(
            id=method, name=name, category=AlgorithmCategory.THRESHOLD,
            description=desc, parameters=params, run_func=run,
        ))
