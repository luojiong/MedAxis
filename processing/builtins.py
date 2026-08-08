"""Registration point for the built-in MedAxis processing algorithms."""

from __future__ import annotations

from .algorithm_registry import AlgorithmRegistry


def register_builtin_algorithms(registry: AlgorithmRegistry | None = None) -> AlgorithmRegistry:
    """Register every built-in algorithm exactly once for the process."""

    registry = registry or AlgorithmRegistry()
    from .filtering.smoothing import register_filtering_algorithms
    from .morphology.binary_ops import register_morphology_algorithms
    from .reconstruction.marching_cubes import register_reconstruction_algorithms
    from .segmentation.threshold import register_segmentation_algorithms
    from .surface.smoothing import register_surface_algorithms
    from .radiomics.features import register_radiomics_algorithms

    for register in (
        register_filtering_algorithms,
        register_morphology_algorithms,
        register_segmentation_algorithms,
        register_reconstruction_algorithms,
        register_surface_algorithms,
        register_radiomics_algorithms,
    ):
        register()
    return registry
