"""
Registration — rigid (Versor), affine, B-spline deformable, Demons variants.
Two-input algorithms: the fixed image is the algorithm input, the moving
image is passed via the ``moving`` parameter.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _registration_v4(transform_name: str):
    def run(fixed_volume, params, progress_callback=None):
        import itk
        moving = params.get("moving", None)
        if moving is None:
            raise ValueError(f"{transform_name}: 'moving' volume parameter is required")

        fixed_img = fixed_volume.to_itk_image()
        moving_img = moving.to_itk_image()

        ImageType = type(fixed_img)
        TransformType = {
            "rigid": itk.VersorRigid3DTransform[itk.D],
            "affine": itk.AffineTransform[itk.D, 3],
            "bspline": itk.BSplineTransform[itk.D, 3],
        }[transform_name]
        transform = TransformType.New()

        if transform_name == "bspline":
            mesh_size = [int(v) for v in params.get("mesh_size", [3, 3, 3])]
            transform.SetTransformDomainOrigin(fixed_img.GetOrigin())
            transform.SetTransformDomainPhysicalDimensions(
                [fixed_img.GetSpacing()[i] * (fixed_img.GetLargestPossibleRegion().GetSize()[i] - 1)
                 for i in range(3)])
            transform.SetTransformDomainMeshSize(mesh_size)
            transform.SetTransformDomainDirection(fixed_img.GetDirection())

        metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
        metric.SetNumberOfHistogramBins(int(params.get("histogram_bins", 32)))

        optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
        optimizer.SetLearningRate(float(params.get("learning_rate", 0.1)))
        optimizer.SetNumberOfIterations(int(params.get("iterations", 100)))
        optimizer.SetMinimumStepLength(1e-4)
        optimizer.SetRelaxationFactor(0.5)

        registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
        registration.SetFixedImage(fixed_img)
        registration.SetMovingImage(moving_img)
        registration.SetInitialTransform(transform)
        registration.SetMetric(metric)
        registration.SetOptimizer(optimizer)
        registration.SetNumberOfLevels(1)
        registration.Update()

        # Apply the transform to the moving image.
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(moving_img)
        resampler.SetTransform(registration.GetTransform())
        resampler.SetSize(fixed_img.GetLargestPossibleRegion().GetSize())
        resampler.SetOutputSpacing(fixed_img.GetSpacing())
        resampler.SetOutputOrigin(fixed_img.GetOrigin())
        resampler.SetOutputDirection(fixed_img.GetDirection())
        resampler.Update()

        stats = {
            "transform": transform_name,
            "iterations": optimizer.GetCurrentIteration(),
            "metric_value": optimizer.GetValue(),
        }
        return AlgorithmResult(
            algorithm_id=transform_name, volume_data=itk.array_from_image(resampler.GetOutput()),
            statistics=stats)

    run.__name__ = f"_register_{transform_name}"
    return run


def _demons(demons_filter, label: str):
    def run(fixed_volume, params, progress_callback=None):
        import itk
        moving = params.get("moving", None)
        if moving is None:
            raise ValueError(f"{label}: 'moving' volume parameter is required")
        fixed_img = fixed_volume.to_itk_image()
        moving_img = moving.to_itk_image()
        iterations = int(params.get("iterations", 20))
        std = float(params.get("standard_deviation", 1.0))

        filter_obj = demons_filter.New()
        filter_obj.SetFixedImage(fixed_img)
        filter_obj.SetMovingImage(moving_img)
        filter_obj.SetNumberOfIterations(iterations)
        filter_obj.SetStandardDeviations(std)
        filter_obj.Update()

        # Warp the moving image with the displacement field.
        warper = itk.WarpImageFilter[type(moving_img), type(moving_img), type(filter_obj.GetOutput())].New()
        warper.SetInput(moving_img)
        warper.SetDisplacementField(filter_obj.GetOutput())
        warper.SetOutputSpacing(fixed_img.GetSpacing())
        warper.SetOutputOrigin(fixed_img.GetOrigin())
        warper.SetOutputDirection(fixed_img.GetDirection())
        warper.Update()

        return AlgorithmResult(
            algorithm_id=label, volume_data=itk.array_from_image(warper.GetOutput()),
            statistics={"transform": label, "iterations": iterations})

    run.__name__ = f"_demons_{label}"
    return run


def _demons_resolved(filt_name: str, label: str):
    """Resolve the ITK demons filter by name at call time (lazy)."""

    def run(fixed_volume, params, progress_callback=None):
        import itk
        filter_obj = getattr(itk, filt_name)
        return _demons(filter_obj, label)(fixed_volume, params, progress_callback)

    run.__name__ = f"_demons_{label}"
    return run


def register_registration_algorithms():
    for name, label in [("rigid", "Rigid (Versor)"), ("affine", "Affine"), ("bspline", "BSpline Deformable")]:
        _registry.register(AlgorithmDefinition(
            id=name, name=label, category=AlgorithmCategory.REGISTRATION,
            description=f"{label} registration (ITKv4, Mattes MI). Requires 'moving' volume.",
            parameters=[
                AlgorithmParameter("moving", "volume", "Moving Volume"),
                AlgorithmParameter("iterations", "int", "Iterations", default=100, min_val=10, max_val=2000),
                AlgorithmParameter("learning_rate", "float", "Learning Rate", default=0.1, min_val=0.001, max_val=1.0),
                AlgorithmParameter("histogram_bins", "int", "Histogram Bins", default=32, min_val=8, max_val=128),
            ],
            run_func=_registration_v4(name),
        ))
    for filt_name, label, desc in [
        ("demons_registration_filter", "demons", "Classic Demons"),
        ("diffeomorphic_demons_registration_filter", "diffeomorphic_demons", "Diffeomorphic Demons"),
        ("fast_symmetric_forces_demons_registration_filter", "fast_symmetric_demons", "Fast Symmetric Demons"),
    ]:
        _registry.register(AlgorithmDefinition(
            id=label, name=desc, category=AlgorithmCategory.REGISTRATION,
            description=f"{desc} registration. Requires 'moving' volume.",
            parameters=[
                AlgorithmParameter("moving", "volume", "Moving Volume"),
                AlgorithmParameter("iterations", "int", "Iterations", default=20, min_val=1, max_val=200),
                AlgorithmParameter("standard_deviation", "float", "Smoothing Std", default=1.0, min_val=0.0, max_val=5.0),
            ],
            # Filter resolved lazily: the pip `itk` binding may name it differently.
            run_func=_demons_resolved(filt_name, label),
        ))
