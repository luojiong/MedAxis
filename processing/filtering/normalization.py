"""
Normalization & intensity mapping algorithms.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _normalize(volume, params, progress_callback=None):
    import itk
    itk_image = volume.to_itk_image()
    filtered = itk.normalize_image_filter(itk_image)
    return AlgorithmResult(algorithm_id="normalize", volume_data=itk.array_from_image(filtered))


def _rescale(volume, params, progress_callback=None):
    import itk
    itk_image = volume.to_itk_image()
    out_min = params.get("output_minimum", 0.0)
    out_max = params.get("output_maximum", 255.0)
    filtered = itk.rescale_intensity_image_filter(
        itk_image, output_minimum=float(out_min), output_maximum=float(out_max))
    return AlgorithmResult(algorithm_id="rescale", volume_data=itk.array_from_image(filtered))


def _cast(volume, params, progress_callback=None):
    """Cast voxel dtype (float32 / int16 / uint8)."""
    import numpy as np
    dtype_name = params.get("dtype", "float32")
    dtype_map = {"float32": np.float32, "int16": np.int16, "uint8": np.uint8, "float64": np.float64}
    dtype = dtype_map.get(dtype_name, np.float32)
    arr = np.asarray(volume.array, dtype=dtype)
    return AlgorithmResult(algorithm_id="cast", volume_data=arr)


def _sharpen(volume, params, progress_callback=None):
    """Laplacian sharpening: img + amount * laplacian(img)."""
    import itk
    import numpy as np
    itk_image = volume.to_itk_image()
    amount = params.get("amount", 1.0)
    lap = itk.laplacian_recursive_gaussian_image_filter(itk_image, sigma=1.0)
    lap_arr = itk.array_from_image(lap)
    base = np.asarray(volume.array, dtype=np.float32)
    sharp = base + float(amount) * lap_arr.astype(np.float32)
    return AlgorithmResult(algorithm_id="sharpen", volume_data=sharp)


def register_normalization_algorithms():
    _registry.register(AlgorithmDefinition(
        id="normalize", name="Normalize", category=AlgorithmCategory.FILTERING,
        description="Zero mean / unit variance normalization.",
        parameters=[], run_func=_normalize,
    ))
    _registry.register(AlgorithmDefinition(
        id="rescale", name="Rescale Intensity", category=AlgorithmCategory.FILTERING,
        description="Linear intensity rescale to [min, max].",
        parameters=[
            AlgorithmParameter("output_minimum", "float", "Output Min", default=0.0),
            AlgorithmParameter("output_maximum", "float", "Output Max", default=255.0),
        ],
        run_func=_rescale,
    ))
    _registry.register(AlgorithmDefinition(
        id="cast", name="Cast Image", category=AlgorithmCategory.FILTERING,
        description="Cast voxel data type.",
        parameters=[
            AlgorithmParameter("dtype", "choice", "Output Dtype",
                               choices=["float32", "float64", "int16", "uint8"], default="float32"),
        ],
        run_func=_cast,
    ))
    _registry.register(AlgorithmDefinition(
        id="sharpen", name="Sharpen", category=AlgorithmCategory.FILTERING,
        description="Laplacian sharpening (img + amount * laplacian).",
        parameters=[AlgorithmParameter("amount", "float", "Amount", default=1.0, min_val=0.0, max_val=5.0)],
        run_func=_sharpen,
    ))
