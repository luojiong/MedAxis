"""
Grayscale morphology — erode / dilate / open / close / top-hat / gradient.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _flat_structuring_element(radius: int, shape: str = "ball"):
    """Build an ITK flat structuring element."""
    import itk
    if shape == "box":
        kernel = itk.FlatStructuringElement[3].Box(radius)
    elif shape == "cross":
        kernel = itk.FlatStructuringElement[3].Cross(radius)
    else:
        kernel = itk.FlatStructuringElement[3].Ball(radius)
    return kernel


def _grayscale_apply(volume, params, kind: str):
    import itk
    radius = int(params.get("radius", 1))
    shape = params.get("kernel_type", "ball")
    kernel = _flat_structuring_element(radius, shape)
    itk_image = volume.to_itk_image()

    if kind == "erode":
        filtered = itk.grayscale_erode_image_filter(itk_image, kernel=kernel)
    elif kind == "dilate":
        filtered = itk.grayscale_dilate_image_filter(itk_image, kernel=kernel)
    elif kind == "open":
        filtered = itk.grayscale_morphological_opening_image_filter(itk_image, kernel=kernel)
    elif kind == "close":
        filtered = itk.grayscale_morphological_closing_image_filter(itk_image, kernel=kernel)
    elif kind == "tophat":
        filtered = itk.white_top_hat_image_filter(itk_image, kernel=kernel)
    elif kind == "bottomhat":
        filtered = itk.black_top_hat_image_filter(itk_image, kernel=kernel)
    elif kind == "gradient":
        filtered = itk.morphological_gradient_image_filter(itk_image, kernel=kernel)
    else:
        raise ValueError(f"unknown grayscale operation: {kind}")
    return AlgorithmResult(algorithm_id=f"grayscale_{kind}", volume_data=itk.array_from_image(filtered))


def _make_grayscale(kind: str, label: str, desc: str):
    def run(volume, params, progress_callback=None):
        return _grayscale_apply(volume, params, kind)
    return run


_KERNEL_PARAMS = [
    AlgorithmParameter("radius", "int", "Kernel Radius", default=1, min_val=1, max_val=10),
    AlgorithmParameter("kernel_type", "choice", "Kernel Shape",
                       choices=["ball", "box", "cross"], default="ball"),
]


def register_grayscale_morphology():
    for kind, label, desc in [
        ("erode", "Grayscale Erosion", "Grayscale erosion with flat structuring element."),
        ("dilate", "Grayscale Dilation", "Grayscale dilation with flat structuring element."),
        ("open", "Grayscale Opening", "Grayscale opening (erode then dilate)."),
        ("close", "Grayscale Closing", "Grayscale closing (dilate then erode)."),
        ("tophat", "Top-Hat Transform", "White top-hat: original minus opening."),
        ("bottomhat", "Bottom-Hat Transform", "Black bottom-hat: closing minus original."),
        ("gradient", "Morphological Gradient", "Dilation minus erosion."),
    ]:
        _registry.register(AlgorithmDefinition(
            id=f"grayscale_{kind}", name=label, category=AlgorithmCategory.MORPHOLOGY,
            description=desc, parameters=list(_KERNEL_PARAMS),
            run_func=_make_grayscale(kind, label, desc),
        ))
