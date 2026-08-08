"""Headless regression tests for the processing registry."""

import numpy as np

from processing.algorithm_registry import (
    AlgorithmCategory,
    AlgorithmDefinition,
    AlgorithmParameter,
    AlgorithmRegistry,
)
from processing.builtins import register_builtin_algorithms


def test_builtin_algorithms_are_registered() -> None:
    registry = AlgorithmRegistry()
    registry._algorithms.clear()
    register_builtin_algorithms(registry)

    assert registry.get("manual_threshold") is not None
    assert registry.get("marching_cubes") is not None
    assert registry.get("first_order_radiomics") is not None


def test_registry_validates_and_coerces_parameters() -> None:
    registry = AlgorithmRegistry()
    observed = {}

    def run(volume, params, progress_callback=None):
        observed.update(params)
        return np.zeros((1, 1, 1), dtype=np.uint8)

    registry.register(AlgorithmDefinition(
        id="validation_probe",
        name="Validation probe",
        category=AlgorithmCategory.SEGMENTATION,
        parameters=[AlgorithmParameter("radius", "int", "Radius", default=2, min_val=1, max_val=4)],
        run_func=run,
    ))

    assert registry.run("validation_probe", object(), {"radius": "3"}).success
    assert observed["radius"] == 3
    assert not registry.run("validation_probe", object(), {"radius": 9}).success
