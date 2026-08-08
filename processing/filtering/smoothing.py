"""
Filtering algorithms — smoothing, edge detection, normalization.
"""
from ..algorithm_registry import AlgorithmRegistry, AlgorithmDefinition, AlgorithmCategory, AlgorithmResult, AlgorithmParameter

_registry = AlgorithmRegistry()


def _gaussian_filter(volume, params, progress_callback=None):
    """Apply Gaussian smoothing."""
    import itk
    sigma = params.get("sigma", 1.0)
    itk_image = volume.to_itk_image()
    filtered = itk.smoothing_recursive_gaussian_image_filter(itk_image, sigma=sigma)
    result_array = itk.array_from_image(filtered)
    return AlgorithmResult(algorithm_id="gaussian", volume_data=result_array)


def _median_filter(volume, params, progress_callback=None):
    import itk
    radius = params.get("radius", 2)
    itk_image = volume.to_itk_image()
    filtered = itk.median_image_filter(itk_image, radius=radius)
    return AlgorithmResult(algorithm_id="median", volume_data=itk.array_from_image(filtered))


def _bilateral_filter(volume, params, progress_callback=None):
    import itk
    domain_sigma = params.get("domain_sigma", 2.0)
    range_sigma = params.get("range_sigma", 50.0)
    itk_image = volume.to_itk_image()
    filtered = itk.bilateral_image_filter(itk_image, domain_sigma=domain_sigma, range_sigma=range_sigma)
    return AlgorithmResult(algorithm_id="bilateral", volume_data=itk.array_from_image(filtered))


def _gradient_magnitude(volume, params, progress_callback=None):
    import itk
    sigma = params.get("sigma", 1.0)
    itk_image = volume.to_itk_image()
    filtered = itk.gradient_magnitude_recursive_gaussian_image_filter(itk_image, sigma=sigma)
    return AlgorithmResult(algorithm_id="gradient_magnitude", volume_data=itk.array_from_image(filtered))


def _laplacian(volume, params, progress_callback=None):
    import itk
    sigma = params.get("sigma", 1.0)
    itk_image = volume.to_itk_image()
    filtered = itk.laplacian_recursive_gaussian_image_filter(itk_image, sigma=sigma)
    return AlgorithmResult(algorithm_id="laplacian", volume_data=itk.array_from_image(filtered))


def register_filtering_algorithms():
    _registry.register(AlgorithmDefinition(
        id="gaussian", name="Gaussian Smoothing", category=AlgorithmCategory.FILTERING,
        description="Apply Gaussian smoothing to reduce noise.",
        parameters=[AlgorithmParameter("sigma", "float", "Sigma", default=1.0, min_val=0.1, max_val=10.0)],
        run_func=_gaussian_filter,
    ))
    _registry.register(AlgorithmDefinition(
        id="median", name="Median Filter", category=AlgorithmCategory.FILTERING,
        parameters=[AlgorithmParameter("radius", "int", "Kernel Radius", default=2, min_val=1, max_val=10)],
        run_func=_median_filter,
    ))
    _registry.register(AlgorithmDefinition(
        id="bilateral", name="Bilateral Filter", category=AlgorithmCategory.FILTERING,
        parameters=[
            AlgorithmParameter("domain_sigma", "float", "Domain Sigma", default=2.0, min_val=0.5, max_val=10.0),
            AlgorithmParameter("range_sigma", "float", "Range Sigma", default=50.0, min_val=1.0, max_val=500.0),
        ],
        run_func=_bilateral_filter,
    ))
    _registry.register(AlgorithmDefinition(
        id="gradient_magnitude", name="Gradient Magnitude", category=AlgorithmCategory.FILTERING,
        parameters=[AlgorithmParameter("sigma", "float", "Sigma", default=1.0, min_val=0.1, max_val=5.0)],
        run_func=_gradient_magnitude,
    ))
    _registry.register(AlgorithmDefinition(
        id="laplacian", name="Laplacian", category=AlgorithmCategory.FILTERING,
        parameters=[AlgorithmParameter("sigma", "float", "Sigma", default=1.0, min_val=0.1, max_val=5.0)],
        run_func=_laplacian,
    ))
