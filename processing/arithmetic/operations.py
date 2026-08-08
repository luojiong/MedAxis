"""
Image arithmetic — add / subtract / multiply / divide / mask / invert.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _binary_arith(op: str, label: str):
    def run(volume, params, progress_callback=None):
        import numpy as np
        a = np.asarray(volume.array, dtype=np.float64)
        operand = params.get("operand", None)
        if operand is None:
            operand = float(params.get("constant", 0.0))
        if isinstance(operand, (int, float)):
            b = float(operand)
        else:
            b = np.asarray(operand.array if hasattr(operand, "array") else operand, dtype=np.float64)
        if op == "add":
            out = a + b
        elif op == "subtract":
            out = a - b
        elif op == "multiply":
            out = a * b
        else:  # divide
            out = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return AlgorithmResult(algorithm_id=f"arith_{op}", volume_data=out.astype(np.float32))
    run.__name__ = f"_arith_{op}"
    return run


def _mask(volume, params, progress_callback=None):
    """Apply a mask volume: keep voxels where mask != 0, else outside value."""
    import numpy as np
    arr = np.asarray(volume.array)
    mask = params.get("mask", None)
    if mask is None:
        raise ValueError("mask: 'mask' volume parameter is required")
    mask_arr = np.asarray(mask.array if hasattr(mask, "array") else mask) != 0
    outside = float(params.get("outside_value", 0.0))
    out = np.where(mask_arr, arr, outside).astype(arr.dtype)
    return AlgorithmResult(algorithm_id="mask", volume_data=out)


def _invert(volume, params, progress_callback=None):
    """Invert intensity: maximum - value."""
    import numpy as np
    arr = np.asarray(volume.array)
    maximum = float(params.get("maximum", arr.max()))
    out = (maximum - arr).astype(arr.dtype)
    return AlgorithmResult(algorithm_id="invert", volume_data=out)


def register_arithmetic_algorithms():
    for op, label, desc in [
        ("add", "Add", "Add a constant or another volume."),
        ("subtract", "Subtract", "Subtract a constant or another volume."),
        ("multiply", "Multiply", "Multiply by a constant or another volume."),
        ("divide", "Divide", "Divide by a constant or another volume (0-safe)."),
    ]:
        _registry.register(AlgorithmDefinition(
            id=f"arith_{op}", name=label, category=AlgorithmCategory.ARITHMETIC,
            description=desc,
            parameters=[
                AlgorithmParameter("operand", "volume", "Other Volume (optional)"),
                AlgorithmParameter("constant", "float", "Constant", default=0.0),
            ],
            run_func=_binary_arith(op, label),
        ))
    _registry.register(AlgorithmDefinition(
        id="mask", name="Mask Image", category=AlgorithmCategory.ARITHMETIC,
        description="Apply a mask volume (keep where mask != 0).",
        parameters=[
            AlgorithmParameter("mask", "volume", "Mask Volume"),
            AlgorithmParameter("outside_value", "float", "Outside Value", default=0.0),
        ],
        run_func=_mask,
    ))
    _registry.register(AlgorithmDefinition(
        id="invert", name="Invert Intensity", category=AlgorithmCategory.ARITHMETIC,
        description="Invert intensity (maximum - value).",
        parameters=[AlgorithmParameter("maximum", "float", "Maximum", default=None)],
        run_func=_invert,
    ))
