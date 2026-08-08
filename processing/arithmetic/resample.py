"""
Resampling & geometry — resample / crop / pad / flip / permute axes.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _resample(volume, params, progress_callback=None):
    """Resample to a new spacing (ITK)."""
    import itk
    spacing = tuple(float(v) for v in params.get("spacing", [1.0, 1.0, 1.0]))
    interpolator = params.get("interpolator", "linear")
    itk_image = volume.to_itk_image()
    resampler = itk.ResampleImageFilter[type(itk_image), type(itk_image)].New()
    resampler.SetInput(itk_image)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize([int(round(volume.dimensions[i] * volume.spacing_mm[i] / spacing[i])) for i in range(3)])
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())
    if interpolator == "nearest":
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[type(itk_image), itk.D].New())
    resampler.Update()
    return AlgorithmResult(algorithm_id="resample", volume_data=itk.array_from_image(resampler.GetOutput()))


def _crop(volume, params, progress_callback=None):
    """Crop to a bounding box [i0,i1,j0,j1,k0,k1]."""
    import numpy as np
    bounds = [int(v) for v in params.get("bounds", [])]
    arr = np.asarray(volume.array)
    if len(bounds) != 6:
        raise ValueError("crop: 'bounds' must be [i0,i1,j0,j1,k0,k1]")
    i0, i1, j0, j1, k0, k1 = bounds
    i1 = min(i1, arr.shape[2])
    j1 = min(j1, arr.shape[1])
    k1 = min(k1, arr.shape[0])
    out = arr[k0:k1, j0:j1, i0:i1]
    return AlgorithmResult(algorithm_id="crop", volume_data=out)


def _pad(volume, params, progress_callback=None):
    """Pad with a constant value."""
    import numpy as np
    pad = [int(v) for v in params.get("pad", [0, 0, 0])]
    value = float(params.get("value", 0.0))
    arr = np.asarray(volume.array)
    out = np.pad(arr, [(pad[2], pad[2]), (pad[1], pad[1]), (pad[0], pad[0])],
                 constant_values=value)
    return AlgorithmResult(algorithm_id="pad", volume_data=out)


def _flip(volume, params, progress_callback=None):
    """Flip along selected axes."""
    import numpy as np
    axes = [bool(v) for v in params.get("axes", [False, False, False])]
    arr = np.asarray(volume.array)
    out = arr
    if axes[0]:
        out = out[:, :, ::-1]
    if axes[1]:
        out = out[:, ::-1, :]
    if axes[2]:
        out = out[::-1, :, :]
    return AlgorithmResult(algorithm_id="flip", volume_data=out)


def _permute(volume, params, progress_callback=None):
    """Permute axes, e.g. order=[2,1,0]."""
    import numpy as np
    order = [int(v) for v in params.get("order", [2, 1, 0])]
    arr = np.asarray(volume.array)
    return AlgorithmResult(algorithm_id="permute", volume_data=np.transpose(arr, order))


def register_resample_algorithms():
    _registry.register(AlgorithmDefinition(
        id="resample", name="Resample", category=AlgorithmCategory.ARITHMETIC,
        description="Resample to a new voxel spacing (ITK).",
        parameters=[
            AlgorithmParameter("spacing", "float3", "Output Spacing (mm)", default=[1.0, 1.0, 1.0]),
            AlgorithmParameter("interpolator", "choice", "Interpolator",
                               choices=["linear", "nearest"], default="linear"),
        ],
        run_func=_resample,
    ))
    _registry.register(AlgorithmDefinition(
        id="crop", name="Crop", category=AlgorithmCategory.ARITHMETIC,
        description="Crop to [i0,i1,j0,j1,k0,k1].",
        parameters=[AlgorithmParameter("bounds", "int6", "Bounds [i0,i1,j0,j1,k0,k1]", default=[])],
        run_func=_crop,
    ))
    _registry.register(AlgorithmDefinition(
        id="pad", name="Pad", category=AlgorithmCategory.ARITHMETIC,
        description="Pad all sides with a constant.",
        parameters=[
            AlgorithmParameter("pad", "int3", "Pad (i,j,k)", default=[0, 0, 0]),
            AlgorithmParameter("value", "float", "Pad Value", default=0.0),
        ],
        run_func=_pad,
    ))
    _registry.register(AlgorithmDefinition(
        id="flip", name="Flip Axes", category=AlgorithmCategory.ARITHMETIC,
        description="Flip along i/j/k axes.",
        parameters=[AlgorithmParameter("axes", "bool3", "Flip (i,j,k)", default=[False, False, False])],
        run_func=_flip,
    ))
    _registry.register(AlgorithmDefinition(
        id="permute", name="Permute Axes", category=AlgorithmCategory.ARITHMETIC,
        description="Transpose axes, e.g. [2,1,0].",
        parameters=[AlgorithmParameter("order", "int3", "Axis Order", default=[2, 1, 0])],
        run_func=_permute,
    ))
