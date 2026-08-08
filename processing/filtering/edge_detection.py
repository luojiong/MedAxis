"""
Edge detection algorithms — Sobel, Canny, gradient-based operators.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _sobel(volume, params, progress_callback=None):
    import itk
    itk_image = volume.to_itk_image()
    filtered = itk.sobel_edge_detection_image_filter(itk_image)
    return AlgorithmResult(algorithm_id="sobel", volume_data=itk.array_from_image(filtered))


def _canny(volume, params, progress_callback=None):
    import itk
    itk_image = volume.to_itk_image()
    lower = params.get("lower_threshold", 20.0)
    upper = params.get("upper_threshold", 60.0)
    variance = params.get("variance", 2.0)
    filtered = itk.canny_edge_detection_image_filter(
        itk_image,
        lower_threshold=float(lower),
        upper_threshold=float(upper),
        variance=float(variance),
    )
    return AlgorithmResult(algorithm_id="canny", volume_data=itk.array_from_image(filtered))


def _zero_crossing(volume, params, progress_callback=None):
    """Zero-crossing edge detector (binary output)."""
    import itk
    itk_image = volume.to_itk_image()
    filtered = itk.zero_crossing_image_filter(itk_image)
    return AlgorithmResult(algorithm_id="zero_crossing", volume_data=itk.array_from_image(filtered))


def register_edge_algorithms():
    _registry.register(AlgorithmDefinition(
        id="sobel", name="Sobel Edge Detection", category=AlgorithmCategory.FILTERING,
        description="Gradient magnitude approximation via Sobel kernels.",
        parameters=[], run_func=_sobel,
    ))
    _registry.register(AlgorithmDefinition(
        id="canny", name="Canny Edge Detection", category=AlgorithmCategory.FILTERING,
        description="Canny edge detector with hysteresis thresholds.",
        parameters=[
            AlgorithmParameter("lower_threshold", "float", "Lower Threshold", default=20.0),
            AlgorithmParameter("upper_threshold", "float", "Upper Threshold", default=60.0),
            AlgorithmParameter("variance", "float", "Variance", default=2.0, min_val=0.1, max_val=10.0),
        ],
        run_func=_canny,
    ))
    _registry.register(AlgorithmDefinition(
        id="zero_crossing", name="Zero-Crossing", category=AlgorithmCategory.FILTERING,
        description="Detect edges at zero crossings of the Laplacian.",
        parameters=[], run_func=_zero_crossing,
    ))
