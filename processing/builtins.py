"""Register all built-in algorithms with the global registry."""

from __future__ import annotations

from typing import Optional

from .algorithm_registry import AlgorithmRegistry


def register_builtin_algorithms(registry: Optional[AlgorithmRegistry] = None) -> AlgorithmRegistry:
    """Register every built-in algorithm family on the global (singleton) registry."""
    if registry is None:
        registry = AlgorithmRegistry()

    # Filtering
    from .filtering.smoothing import register_filtering_algorithms, _register_extra_filtering
    from .filtering.edge_detection import register_edge_algorithms
    from .filtering.normalization import register_normalization_algorithms

    # Morphology
    from .morphology.binary_ops import register_morphology_algorithms
    from .morphology.grayscale_ops import register_grayscale_morphology
    from .morphology.hole_filling import register_morphology_extra

    # Segmentation & thresholds
    from .segmentation.threshold import register_segmentation_algorithms, register_histogram_thresholds
    from .segmentation.advanced import register_advanced_segmentation

    # Reconstruction & surface
    from .reconstruction.marching_cubes import register_reconstruction_algorithms
    from .reconstruction.advanced import register_reconstruction_extra
    from .surface.smoothing import register_surface_algorithms
    from .surface.advanced import register_surface_extra

    # Registration
    from .registration.registration import register_registration_algorithms

    # Arithmetic & enhancement
    from .arithmetic.operations import register_arithmetic_algorithms
    from .arithmetic.resample import register_resample_algorithms
    from .enhancement.enhancement import register_enhancement_algorithms

    # Radiomics
    from .radiomics.features import register_radiomics_algorithms, register_full_radiomics

    for register in (
        register_filtering_algorithms,
        _register_extra_filtering,
        register_edge_algorithms,
        register_normalization_algorithms,
        register_morphology_algorithms,
        register_grayscale_morphology,
        register_morphology_extra,
        register_segmentation_algorithms,
        register_histogram_thresholds,
        register_advanced_segmentation,
        register_reconstruction_algorithms,
        register_reconstruction_extra,
        register_surface_algorithms,
        register_surface_extra,
        register_registration_algorithms,
        register_arithmetic_algorithms,
        register_resample_algorithms,
        register_enhancement_algorithms,
        register_radiomics_algorithms,
        register_full_radiomics,
    ):
        register()

    return registry


__all__ = ["register_builtin_algorithms"]
