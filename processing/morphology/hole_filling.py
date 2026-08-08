"""
Binary morphology extras — top-hat, gradient, hole filling, skeleton,
distance map, connected components.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _binary_tophat(volume, params, progress_callback=None):
    import itk
    import numpy as np
    radius = int(params.get("radius", 1))
    kernel = itk.FlatStructuringElement[3].Ball(radius)
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    itk_image = itk.image_from_array(arr)
    if params.get("kind", "tophat") == "bottomhat":
        filtered = itk.binary_black_top_hat_image_filter(itk_image, kernel=kernel)
    else:
        filtered = itk.binary_white_top_hat_image_filter(itk_image, kernel=kernel)
    return AlgorithmResult(algorithm_id="binary_tophat", volume_data=itk.array_from_image(filtered))


def _binary_gradient(volume, params, progress_callback=None):
    import itk
    import numpy as np
    radius = int(params.get("radius", 1))
    kernel = itk.FlatStructuringElement[3].Ball(radius)
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    itk_image = itk.image_from_array(arr)
    filtered = itk.binary_morphological_gradient_image_filter(itk_image, kernel=kernel)
    return AlgorithmResult(algorithm_id="binary_gradient", volume_data=itk.array_from_image(filtered))


def _hole_filling(volume, params, progress_callback=None):
    """Fill enclosed background holes inside foreground regions."""
    import itk
    import numpy as np
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    itk_image = itk.image_from_array(arr)
    fully_connected = bool(params.get("fully_connected", False))
    filtered = itk.binary_fillhole_image_filter(itk_image, fully_connected=fully_connected)
    return AlgorithmResult(algorithm_id="hole_filling", volume_data=itk.array_from_image(filtered))


def _thinning(volume, params, progress_callback=None):
    """Binary thinning (skeletonization) — ITK 5.4 exposes it via scikit-image."""
    import numpy as np
    from skimage.morphology import skeletonize
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    # skimage skeletonize handles 2D and 3D binary volumes.
    skel = skeletonize(arr > 0).astype(np.uint8)
    return AlgorithmResult(algorithm_id="thinning", volume_data=skel)


def _distance_map(volume, params, progress_callback=None):
    """Signed distance map (Maurer) of a binary volume."""
    import itk
    import numpy as np
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    itk_image = itk.image_from_array(arr)
    inside_positive = bool(params.get("inside_positive", True))
    filtered = itk.signed_maurer_distance_map_image_filter(
        itk_image, inside_positive=inside_positive, squared_distance=False)
    return AlgorithmResult(algorithm_id="distance_map", volume_data=itk.array_from_image(filtered))


def _connected_components(volume, params, progress_callback=None):
    """Label connected components; optionally keep only the largest."""
    import itk
    import numpy as np
    arr = (np.asarray(volume.array) > 0).astype(np.uint8)
    itk_image = itk.image_from_array(arr)
    fully_connected = bool(params.get("fully_connected", False))
    labeled = itk.connected_component_image_filter(
        itk_image, fully_connected=fully_connected)
    # Relabel sequentially.
    relabel = itk.relabel_component_image_filter(labeled)
    labels = itk.array_from_image(relabel).astype(np.int32)

    keep_largest = bool(params.get("keep_largest", False))
    if keep_largest and labels.max() > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        labels = (labels == largest).astype(np.uint8)
    return AlgorithmResult(algorithm_id="connected_components", label_data=labels)


def _label_statistics(volume, params, progress_callback=None):
    """Per-component volume statistics (voxel count / mm^3)."""
    import numpy as np
    arr = np.asarray(volume.array)
    if arr.ndim != 3:
        raise ValueError("label_statistics requires a 3D label volume")
    spacing = getattr(volume, "spacing_mm", (1.0, 1.0, 1.0))
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    counts = np.bincount(arr.ravel())
    stats = {}
    for label_value in np.unique(arr):
        if label_value == 0:
            continue
        n = int(counts[label_value])
        stats[int(label_value)] = {
            "voxels": n,
            "volume_mm3": round(n * voxel_vol, 3),
            "volume_ml": round(n * voxel_vol / 1000.0, 4),
        }
    return AlgorithmResult(
        algorithm_id="label_statistics", statistics={"labels": stats, "voxel_volume_mm3": voxel_vol})


def register_morphology_extra():
    _registry.register(AlgorithmDefinition(
        id="binary_tophat", name="Binary Top-Hat", category=AlgorithmCategory.MORPHOLOGY,
        description="White/black top-hat on a binary volume.",
        parameters=[
            AlgorithmParameter("radius", "int", "Kernel Radius", default=1, min_val=1, max_val=10),
            AlgorithmParameter("kind", "choice", "Transform", choices=["tophat", "bottomhat"], default="tophat"),
        ],
        run_func=_binary_tophat,
    ))
    _registry.register(AlgorithmDefinition(
        id="binary_gradient", name="Binary Morphological Gradient", category=AlgorithmCategory.MORPHOLOGY,
        description="Dilation minus erosion on a binary volume.",
        parameters=[AlgorithmParameter("radius", "int", "Kernel Radius", default=1, min_val=1, max_val=10)],
        run_func=_binary_gradient,
    ))
    _registry.register(AlgorithmDefinition(
        id="hole_filling", name="Hole Filling", category=AlgorithmCategory.MORPHOLOGY,
        description="Fill enclosed background holes inside foreground regions.",
        parameters=[AlgorithmParameter("fully_connected", "bool", "Fully Connected", default=False)],
        run_func=_hole_filling,
    ))
    _registry.register(AlgorithmDefinition(
        id="thinning", name="Skeletonization (Thinning)", category=AlgorithmCategory.MORPHOLOGY,
        description="Morphological skeleton of a binary volume.",
        parameters=[], run_func=_thinning,
    ))
    _registry.register(AlgorithmDefinition(
        id="distance_map", name="Signed Distance Map", category=AlgorithmCategory.MORPHOLOGY,
        description="Maurer signed distance transform of a binary volume.",
        parameters=[AlgorithmParameter("inside_positive", "bool", "Inside Positive", default=True)],
        run_func=_distance_map,
    ))
    _registry.register(AlgorithmDefinition(
        id="connected_components", name="Connected Components", category=AlgorithmCategory.MORPHOLOGY,
        description="Label connected components; optionally keep the largest.",
        parameters=[
            AlgorithmParameter("fully_connected", "bool", "Fully Connected", default=False),
            AlgorithmParameter("keep_largest", "bool", "Keep Largest Only", default=False),
        ],
        run_func=_connected_components,
    ))
    _registry.register(AlgorithmDefinition(
        id="label_statistics", name="Label Statistics", category=AlgorithmCategory.MEASUREMENT,
        description="Per-component voxel count and physical volume.",
        parameters=[], run_func=_label_statistics,
    ))
