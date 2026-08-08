"""
Morphological operations — binary and grayscale.
"""
from ..algorithm_registry import AlgorithmRegistry, AlgorithmDefinition, AlgorithmCategory, AlgorithmResult, AlgorithmParameter

_registry = AlgorithmRegistry()


def _binary_morphology(label_data, params, progress_callback=None):
    """Binary morphology: erode / dilate / open / close."""
    import itk
    import numpy as np
    operation = params.get("operation", "dilate")
    radius = params.get("radius", 2)
    kernel_type = params.get("kernel_type", "ball")

    itk_label = itk.image_from_array(label_data.array.astype(np.uint8))

    if operation == "erode":
        result = itk.binary_erode_image_filter(itk_label, radius=radius, kernel_type=kernel_type)
    elif operation == "dilate":
        result = itk.binary_dilate_image_filter(itk_label, radius=radius, kernel_type=kernel_type)
    elif operation == "open":
        result = itk.binary_morphological_opening_image_filter(itk_label, radius=radius, kernel_type=kernel_type)
    elif operation == "close":
        result = itk.binary_morphological_closing_image_filter(itk_label, radius=radius, kernel_type=kernel_type)
    else:
        return AlgorithmResult(algorithm_id="binary_morphology", success=False, error=f"Unknown operation: {operation}")

    return AlgorithmResult(algorithm_id="binary_morphology", label_data=itk.array_from_image(result))


def _grayscale_morphology(volume, params, progress_callback=None):
    """Grayscale morphology."""
    import itk
    operation = params.get("operation", "dilate")
    radius = params.get("radius", 2)
    itk_image = volume.to_itk_image()

    if operation == "erode":
        result = itk.grayscale_erode_image_filter(itk_image, radius=radius)
    elif operation == "dilate":
        result = itk.grayscale_dilate_image_filter(itk_image, radius=radius)
    elif operation == "open":
        result = itk.grayscale_morphological_opening_image_filter(itk_image, radius=radius)
    elif operation == "close":
        result = itk.grayscale_morphological_closing_image_filter(itk_image, radius=radius)
    else:
        return AlgorithmResult(algorithm_id="grayscale_morphology", success=False, error=f"Unknown operation: {operation}")

    return AlgorithmResult(algorithm_id="grayscale_morphology", label_data=itk.array_from_image(result))


def _connected_components(label_data, params, progress_callback=None):
    """Connected component labeling."""
    import itk
    import numpy as np
    itk_label = itk.image_from_array(label_data.array.astype(np.uint8))
    result = itk.connected_component_image_filter(itk_label)
    return AlgorithmResult(algorithm_id="connected_components", label_data=itk.array_from_image(result))


def _distance_map(label_data, params, progress_callback=None):
    """Signed distance map."""
    import itk
    import numpy as np
    itk_label = itk.image_from_array(label_data.array.astype(np.uint8))
    result = itk.signed_maurer_distance_map_image_filter(itk_label)
    return AlgorithmResult(algorithm_id="distance_map", label_data=itk.array_from_image(result))


def register_morphology_algorithms():
    _registry.register(AlgorithmDefinition(
        id="binary_morphology", name="Binary Morphology", category=AlgorithmCategory.MORPHOLOGY,
        parameters=[
            AlgorithmParameter("operation", "choice", "Operation", choices=["erode", "dilate", "open", "close"], default="dilate"),
            AlgorithmParameter("radius", "int", "Kernel Radius", default=2, min_val=1, max_val=20),
            AlgorithmParameter("kernel_type", "choice", "Kernel Type", choices=["ball", "box", "cross"], default="ball"),
        ],
        run_func=_binary_morphology,
        input_parameter="label_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="grayscale_morphology", name="Grayscale Morphology", category=AlgorithmCategory.MORPHOLOGY,
        parameters=[
            AlgorithmParameter("operation", "choice", "Operation", choices=["erode", "dilate", "open", "close"], default="dilate"),
            AlgorithmParameter("radius", "int", "Kernel Radius", default=2, min_val=1, max_val=20),
        ],
        run_func=_grayscale_morphology,
    ))
    _registry.register(AlgorithmDefinition(
        id="connected_components", name="Connected Components", category=AlgorithmCategory.MORPHOLOGY,
        run_func=_connected_components,
        input_parameter="label_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="distance_map", name="Distance Map", category=AlgorithmCategory.MORPHOLOGY,
        run_func=_distance_map,
        input_parameter="label_data",
    ))
