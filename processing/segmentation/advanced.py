"""
Advanced segmentation — confidence/neighborhood connected, level sets,
fast marching, clustering.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _confidence_connected(volume, params, progress_callback=None):
    """Confidence-connected region growing from a seed point."""
    import itk
    seed = tuple(int(v) for v in params.get("seed", [0, 0, 0]))
    itk_image = volume.to_itk_image()
    seed_index = itk.Index[3]()
    seed_index[0], seed_index[1], seed_index[2] = seed[2], seed[1], seed[0]
    multiplier = params.get("multiplier", 2.0)
    iterations = params.get("iterations", 5)
    result = itk.confidence_connected_image_filter(
        itk_image, seed=seed_index,
        multiplier=float(multiplier), number_of_iterations=int(iterations))
    return AlgorithmResult(algorithm_id="confidence_connected", label_data=itk.array_from_image(result))


def _neighborhood_connected(volume, params, progress_callback=None):
    """Neighborhood-connected region growing (fixed intensity window)."""
    import itk
    seed = tuple(int(v) for v in params.get("seed", [0, 0, 0]))
    itk_image = volume.to_itk_image()
    seed_index = itk.Index[3]()
    seed_index[0], seed_index[1], seed_index[2] = seed[2], seed[1], seed[0]
    lower = params.get("lower", 0)
    upper = params.get("upper", 255)
    radius = int(params.get("radius", 1))
    result = itk.neighborhood_connected_image_filter(
        itk_image, seed=seed_index,
        lower=float(lower), upper=float(upper), radius=radius)
    return AlgorithmResult(algorithm_id="neighborhood_connected", label_data=itk.array_from_image(result))


def _level_set(method: str):
    def run(volume, params, progress_callback=None):
        import itk
        import numpy as np
        itk_image = volume.to_itk_image()
        iterations = int(params.get("iterations", 100))
        curvature = float(params.get("curvature_scaling", 1.0))
        propagation = float(params.get("propagation_scaling", 1.0))

        # Level-set filters need a seed (fast marching) or an initial level set.
        # We build the initial level set from a threshold of the image.
        seed = tuple(int(v) for v in params.get("seed", []))
        if not seed:
            raise ValueError(f"{method}: a seed point is required")

        # 1. Fast marching initial surface from the seed.
        seed_index = itk.Index[3]()
        seed_index[0], seed_index[1], seed_index[2] = seed[2], seed[1], seed[0]
        stopping_time = float(params.get("stopping_time", 50.0))
        fm = itk.fast_marching_image_filter(
            itk_image, seed_value=seed_index, stopping_value=stopping_time)

        # 2. Run the requested level-set evolution.
        if method == "geodesic_active_contour":
            filtered = itk.geodesic_active_contour_level_set_image_filter(
                itk_image, feature_image=itk_image,
                initial_level_set=fm, number_of_iterations=iterations,
                propagation_scaling=propagation, curvature_scaling=curvature)
        elif method == "shape_detection":
            filtered = itk.shape_detection_level_set_image_filter(
                itk_image, feature_image=itk_image,
                initial_level_set=fm, number_of_iterations=iterations,
                propagation_scaling=propagation, curvature_scaling=curvature)
        elif method == "threshold_segmentation":
            lower = float(params.get("lower_threshold", 0))
            upper = float(params.get("upper_threshold", 255))
            filtered = itk.threshold_segmentation_level_set_image_filter(
                itk_image, feature_image=itk_image,
                initial_level_set=fm, number_of_iterations=iterations,
                lower_threshold=lower, upper_threshold=upper,
                curvature_scaling=curvature, propagation_scaling=propagation)
        elif method == "laplacian_segmentation":
            filtered = itk.laplacian_segmentation_level_set_image_filter(
                itk_image, feature_image=itk_image,
                initial_level_set=fm, number_of_iterations=iterations,
                curvature_scaling=curvature, propagation_scaling=propagation)
        else:
            raise ValueError(f"unknown level set: {method}")

        label = (itk.array_from_image(filtered) > 0).astype(np.uint8)
        return AlgorithmResult(algorithm_id=method, label_data=label)

    run.__name__ = f"_{method}"
    return run


def _fast_marching(volume, params, progress_callback=None):
    """Fast marching segmentation from a seed point."""
    import itk
    import numpy as np
    seed = tuple(int(v) for v in params.get("seed", []))
    if not seed:
        raise ValueError("fast_marching: a seed point is required")
    itk_image = volume.to_itk_image()
    seed_index = itk.Index[3]()
    seed_index[0], seed_index[1], seed_index[2] = seed[2], seed[1], seed[0]
    stopping_time = float(params.get("stopping_time", 50.0))
    fm = itk.fast_marching_image_filter(
        itk_image, seed_value=seed_index, stopping_value=stopping_time)
    label = (itk.array_from_image(fm) > 0).astype(np.uint8)
    return AlgorithmResult(algorithm_id="fast_marching", label_data=label)


def _kmeans(volume, params, progress_callback=None):
    """Scalar K-means clustering into N classes."""
    import itk
    num_clusters = int(params.get("num_clusters", 3))
    max_iterations = int(params.get("max_iterations", 100))
    itk_image = volume.to_itk_image()
    result = itk.scalar_image_kmeans_image_filter(
        itk_image, number_of_initial_means=num_clusters,
        maximum_number_of_iterations=max_iterations)
    return AlgorithmResult(algorithm_id="kmeans", label_data=itk.array_from_image(result))


def _fcm(volume, params, progress_callback=None):
    """Fuzzy C-means via scikit-image (numpy implementation of FCM)."""
    import numpy as np
    num_clusters = int(params.get("num_clusters", 3))
    fuzziness = float(params.get("fuzziness", 2.0))
    max_iterations = int(params.get("max_iterations", 100))
    arr = np.asarray(volume.array, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("fcm: empty volume")

    # Standard FCM on intensity values.
    centers = np.linspace(arr.min(), arr.max(), num_clusters)
    u = np.zeros((num_clusters, arr.size))
    m = fuzziness
    for _ in range(max_iterations):
        dist = np.abs(arr[None, :] - centers[:, None])
        dist = np.maximum(dist, 1e-12)
        # Update memberships.
        inv = 1.0 / dist
        u = inv ** (2.0 / (m - 1.0))
        u = u / u.sum(axis=0, keepdims=True)
        new_centers = (u ** m) @ arr / (u ** m).sum(axis=1)
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    labels = np.argmax(u, axis=0).reshape(volume.array.shape).astype(np.uint8)
    return AlgorithmResult(algorithm_id="fcm", label_data=labels)


def _island_remove(volume, params, progress_callback=None):
    """Remove all but the largest connected component of a label volume."""
    import numpy as np
    from scipy import ndimage
    arr = np.asarray(volume.array)
    labeled, n = ndimage.label(arr > 0)
    if n == 0:
        return AlgorithmResult(algorithm_id="island_remove", label_data=arr.astype(np.uint8))
    counts = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, n + 1))
    keep = int(np.argmax(counts)) + 1
    out = (labeled == keep).astype(np.uint8)
    return AlgorithmResult(algorithm_id="island_remove", label_data=out)


_SEED_PARAMS = [
    AlgorithmParameter("seed", "point3d", "Seed Point (i,j,k)"),
]


def register_advanced_segmentation():
    _registry.register(AlgorithmDefinition(
        id="confidence_connected", name="Confidence Connected", category=AlgorithmCategory.SEGMENTATION,
        description="Region growing with statistical confidence criteria.",
        parameters=_SEED_PARAMS + [
            AlgorithmParameter("multiplier", "float", "Multiplier", default=2.0, min_val=0.5, max_val=10.0),
            AlgorithmParameter("iterations", "int", "Iterations", default=5, min_val=1, max_val=100),
        ],
        run_func=_confidence_connected,
    ))
    _registry.register(AlgorithmDefinition(
        id="neighborhood_connected", name="Neighborhood Connected", category=AlgorithmCategory.SEGMENTATION,
        description="Region growing within a fixed intensity window.",
        parameters=_SEED_PARAMS + [
            AlgorithmParameter("lower", "float", "Lower Threshold", default=0),
            AlgorithmParameter("upper", "float", "Upper Threshold", default=255),
            AlgorithmParameter("radius", "int", "Neighborhood Radius", default=1, min_val=1, max_val=5),
        ],
        run_func=_neighborhood_connected,
    ))
    for method, name, desc in [
        ("geodesic_active_contour", "Geodesic Active Contour", "Geodesic level-set evolution."),
        ("shape_detection", "Shape Detection Level Set", "Shape-detection level-set evolution."),
        ("threshold_segmentation", "Threshold Level Set", "Threshold-guided level-set evolution."),
        ("laplacian_segmentation", "Laplacian Level Set", "Laplacian-driven level-set evolution."),
    ]:
        _registry.register(AlgorithmDefinition(
            id=method, name=name, category=AlgorithmCategory.SEGMENTATION,
            description=desc,
            parameters=_SEED_PARAMS + [
                AlgorithmParameter("iterations", "int", "Iterations", default=100, min_val=1, max_val=2000),
                AlgorithmParameter("curvature_scaling", "float", "Curvature Scaling", default=1.0, min_val=0.0, max_val=5.0),
                AlgorithmParameter("propagation_scaling", "float", "Propagation Scaling", default=1.0, min_val=0.0, max_val=5.0),
                AlgorithmParameter("stopping_time", "float", "Stopping Time", default=50.0, min_val=1.0, max_val=500.0),
            ],
            run_func=_level_set(method),
        ))
    _registry.register(AlgorithmDefinition(
        id="fast_marching", name="Fast Marching", category=AlgorithmCategory.SEGMENTATION,
        description="Fast marching front propagation from a seed.",
        parameters=_SEED_PARAMS + [
            AlgorithmParameter("stopping_time", "float", "Stopping Time", default=50.0, min_val=1.0, max_val=500.0),
        ],
        run_func=_fast_marching,
    ))
    _registry.register(AlgorithmDefinition(
        id="kmeans", name="K-Means Clustering", category=AlgorithmCategory.SEGMENTATION,
        description="Scalar K-means intensity clustering.",
        parameters=[
            AlgorithmParameter("num_clusters", "int", "Clusters", default=3, min_val=2, max_val=10),
            AlgorithmParameter("max_iterations", "int", "Max Iterations", default=100, min_val=10, max_val=1000),
        ],
        run_func=_kmeans,
    ))
    _registry.register(AlgorithmDefinition(
        id="fcm", name="Fuzzy C-Means", category=AlgorithmCategory.SEGMENTATION,
        description="Fuzzy C-means intensity clustering.",
        parameters=[
            AlgorithmParameter("num_clusters", "int", "Clusters", default=3, min_val=2, max_val=10),
            AlgorithmParameter("fuzziness", "float", "Fuzziness (m)", default=2.0, min_val=1.1, max_val=5.0),
            AlgorithmParameter("max_iterations", "int", "Max Iterations", default=100, min_val=10, max_val=1000),
        ],
        run_func=_fcm,
    ))
    _registry.register(AlgorithmDefinition(
        id="island_remove", name="Island Remove (Keep Largest)", category=AlgorithmCategory.SEGMENTATION,
        description="Keep the largest connected component of a label volume.",
        parameters=[], run_func=_island_remove,
    ))
