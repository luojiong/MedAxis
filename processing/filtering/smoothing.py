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


def _anisotropic_diffusion(volume, params, progress_callback=None):
    """Perona-Malik anisotropic diffusion (edge-preserving smoothing)."""
    import itk
    itk_image = volume.to_itk_image()
    iterations = params.get("iterations", 5)
    time_step = params.get("time_step", 0.0625)
    conductance = params.get("conductance", 3.0)
    filtered = itk.gradient_anisotropic_diffusion_image_filter(
        itk_image,
        number_of_iterations=int(iterations),
        time_step=float(time_step),
        conductance=float(conductance),
    )
    return AlgorithmResult(algorithm_id="anisotropic_diffusion", volume_data=itk.array_from_image(filtered))


def _nl_means(volume, params, progress_callback=None):
    """Non-local means denoising (scikit-image)."""
    import numpy as np
    from skimage.restoration import denoise_nl_means
    arr = np.asarray(volume.array, dtype=np.float64)
    patch_radius = params.get("patch_radius", 1)
    search_radius = params.get("search_radius", 3)
    h = params.get("h", 0.1)
    denoised = denoise_nl_means(
        arr, h=float(h), patch_size=2 * int(patch_radius) + 1,
        patch_distance=2 * int(search_radius) + 1, fast_mode=True,
    )
    return AlgorithmResult(algorithm_id="nl_means", volume_data=denoised)


def _mean_filter(volume, params, progress_callback=None):
    import itk
    radius = params.get("radius", 2)
    itk_image = volume.to_itk_image()
    filtered = itk.mean_image_filter(itk_image, radius=int(radius))
    return AlgorithmResult(algorithm_id="mean", volume_data=itk.array_from_image(filtered))


def _discrete_gaussian(volume, params, progress_callback=None):
    import itk
    variance = params.get("variance", 1.0)
    itk_image = volume.to_itk_image()
    filtered = itk.discrete_gaussian_image_filter(itk_image, variance=float(variance))
    return AlgorithmResult(algorithm_id="discrete_gaussian", volume_data=itk.array_from_image(filtered))


def _register_extra_filtering():
    _registry.register(AlgorithmDefinition(
        id="anisotropic_diffusion", name="Anisotropic Diffusion", category=AlgorithmCategory.FILTERING,
        description="Perona-Malik edge-preserving anisotropic diffusion.",
        parameters=[
            AlgorithmParameter("iterations", "int", "Iterations", default=5, min_val=1, max_val=100),
            AlgorithmParameter("time_step", "float", "Time Step", default=0.0625, min_val=0.001, max_val=0.5),
            AlgorithmParameter("conductance", "float", "Conductance", default=3.0, min_val=0.1, max_val=10.0),
        ],
        run_func=_anisotropic_diffusion,
    ))
    _registry.register(AlgorithmDefinition(
        id="nl_means", name="NL-Means Denoising", category=AlgorithmCategory.FILTERING,
        description="Non-local means denoising (scikit-image).",
        parameters=[
            AlgorithmParameter("patch_radius", "int", "Patch Radius", default=1, min_val=0, max_val=4),
            AlgorithmParameter("search_radius", "int", "Search Radius", default=3, min_val=1, max_val=10),
            AlgorithmParameter("h", "float", "Filter Strength", default=0.1, min_val=0.01, max_val=1.0),
        ],
        run_func=_nl_means,
    ))
    _registry.register(AlgorithmDefinition(
        id="mean", name="Mean Filter", category=AlgorithmCategory.FILTERING,
        description="Local mean (box) filter.",
        parameters=[AlgorithmParameter("radius", "int", "Kernel Radius", default=2, min_val=1, max_val=10)],
        run_func=_mean_filter,
    ))
    _registry.register(AlgorithmDefinition(
        id="discrete_gaussian", name="Discrete Gaussian", category=AlgorithmCategory.FILTERING,
        description="Discrete Gaussian smoothing by variance.",
        parameters=[AlgorithmParameter("variance", "float", "Variance", default=1.0, min_val=0.1, max_val=20.0)],
        run_func=_discrete_gaussian,
    ))
