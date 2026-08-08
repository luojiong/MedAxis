/*
 * Native ITK-backed operations: advanced segmentation and registration.
 */

#include "native_ops.h"
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <itkAffineTransform.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkBSplineTransform.h>
#include <itkConfidenceConnectedImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkDiffeomorphicDemonsRegistrationFilter.h>
#include <itkDemonsRegistrationFilter.h>
#include <itkFastMarchingImageFilter.h>
#include <itkFastSymmetricForcesDemonsRegistrationFilter.h>
#include <itkGeodesicActiveContourLevelSetImageFilter.h>
#include <itkImage.h>
#include <itkImageRegistrationMethodv4.h>
#include <itkImageRegionConstIterator.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkLaplacianSegmentationLevelSetImageFilter.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkNeighborhoodConnectedImageFilter.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkResampleImageFilter.h>
#include <itkScalarImageKmeansImageFilter.h>
#include <itkShapeDetectionLevelSetImageFilter.h>
#include <itkThresholdSegmentationLevelSetImageFilter.h>
#include <itkVersorRigid3DTransform.h>
#include <itkWarpImageFilter.h>
#include <itkWatershedImageFilter.h>

namespace medaxis
{
namespace native_ops
{

namespace
{

using ImageF = itk::Image<float, 3>;
using ImageU8 = itk::Image<std::uint8_t, 3>;
using ImportF = itk::ImportImageFilter<float, 3>;

inline float param_float(const py::dict& params, const char* key, float def)
{
    if (!params.contains(key)) return def;
    return static_cast<float>(py::cast<double>(params[key]));
}

inline int param_int(const py::dict& params, const char* key, int def)
{
    if (!params.contains(key)) return def;
    return static_cast<int>(py::cast<long long>(params[key]));
}

ImageF::Pointer import_image(const float* data, py::ssize_t nk, py::ssize_t nj, py::ssize_t ni)
{
    auto importer = ImportF::New();
    ImageF::SizeType size;
    size[0] = static_cast<ImageF::SizeType::SizeValueType>(ni);
    size[1] = static_cast<ImageF::SizeType::SizeValueType>(nj);
    size[2] = static_cast<ImageF::SizeType::SizeValueType>(nk);
    ImageF::RegionType region;
    region.SetSize(size);
    importer->SetRegion(region);
    importer->SetImportPointer(const_cast<float*>(data),
                               static_cast<unsigned long long>(ni * nj * nk), false);
    importer->Update();
    return importer->GetOutput();
}

py::array_t<std::uint8_t> export_label(ImageU8::Pointer image)
{
    const auto size = image->GetLargestPossibleRegion().GetSize();
    const py::ssize_t nk = static_cast<py::ssize_t>(size[2]);
    const py::ssize_t nj = static_cast<py::ssize_t>(size[1]);
    const py::ssize_t ni = static_cast<py::ssize_t>(size[0]);
    py::array_t<std::uint8_t> out({nk, nj, ni});
    auto* dst = out.mutable_data();
    itk::ImageRegionConstIterator<ImageU8> it(image, image->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        const auto index = it.GetIndex();
        const std::size_t offset =
            (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
             static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
            static_cast<std::size_t>(index[0]);
        dst[offset] = it.Get();
    }
    return out;
}

py::array_t<float> export_float(ImageF::Pointer image)
{
    const auto size = image->GetLargestPossibleRegion().GetSize();
    const py::ssize_t nk = static_cast<py::ssize_t>(size[2]);
    const py::ssize_t nj = static_cast<py::ssize_t>(size[1]);
    const py::ssize_t ni = static_cast<py::ssize_t>(size[0]);
    py::array_t<float> out({nk, nj, ni});
    auto* dst = out.mutable_data();
    itk::ImageRegionConstIterator<ImageF> it(image, image->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        const auto index = it.GetIndex();
        const std::size_t offset =
            (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
             static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
            static_cast<std::size_t>(index[0]);
        dst[offset] = it.Get();
    }
    return out;
}

ImageF::IndexType seed_index(const py::dict& params)
{
    const std::vector<int> seed = py::cast<std::vector<int>>(params["seed"]);
    if (seed.size() != 3)
    {
        throw std::invalid_argument("a seed point [i, j, k] is required");
    }
    ImageF::IndexType index;
    index[0] = seed[0];
    index[1] = seed[1];
    index[2] = seed[2];
    return index;
}

/// Fast marching arrival-time image used as the initial level set.
ImageF::Pointer fast_marching_init(ImageF::Pointer input, const py::dict& params,
                                   ImageF::IndexType seed)
{
    using FM = itk::FastMarchingImageFilter<ImageF, ImageF>;
    auto fm = FM::New();
    fm->SetInput(input);
    fm->SetStoppingValue(param_float(params, "stopping_time", 50.0f));
    typename FM::NodeContainer::Pointer seeds = FM::NodeContainer::New();
    typename FM::NodeType node;
    node.SetValue(0.0);
    node.SetIndex(seed);
    seeds->Initialize();
    seeds->InsertElement(0, node);
    fm->SetTrialPoints(seeds);
    fm->Update();
    return fm->GetOutput();
}

} // namespace

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

py::array_t<std::uint8_t> segment_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind, const py::dict& params)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("segment_3d expects a 3-D array (z, y, x)");
    }
    const py::ssize_t nk = values.shape(0), nj = values.shape(1), ni = values.shape(2);
    ImageF::Pointer input = import_image(values.data(), nk, nj, ni);
    ImageF::SizeType size = input->GetLargestPossibleRegion().GetSize();

    // Threshold-style segmentations (uint8).
    if (kind == "connected_threshold" || kind == "confidence_connected" || kind == "neighborhood_connected")
    {
        const ImageF::IndexType seed = seed_index(params);
        ImageU8::Pointer label;
        if (kind == "connected_threshold")
        {
            using FilterT = itk::ConnectedThresholdImageFilter<ImageF, ImageU8>;
            auto f = FilterT::New();
            f->SetInput(input);
            f->AddSeed(seed);
            f->SetLower(param_float(params, "lower", 0.0f));
            f->SetUpper(param_float(params, "upper", 255.0f));
            f->Update();
            label = f->GetOutput();
        }
        else if (kind == "confidence_connected")
        {
            using FilterT = itk::ConfidenceConnectedImageFilter<ImageF, ImageU8>;
            auto f = FilterT::New();
            f->SetInput(input);
            f->AddSeed(seed);
            f->SetMultiplier(param_float(params, "multiplier", 2.0f));
            f->SetNumberOfIterations(param_int(params, "iterations", 5));
            f->Update();
            label = f->GetOutput();
        }
        else
        {
            using FilterT = itk::NeighborhoodConnectedImageFilter<ImageF, ImageU8>;
            auto f = FilterT::New();
            f->SetInput(input);
            f->AddSeed(seed);
            f->SetLower(param_float(params, "lower", 0.0f));
            f->SetUpper(param_float(params, "upper", 255.0f));
            ImageF::SizeType radius;
            radius.Fill(param_int(params, "radius", 1));
            f->SetRadius(radius);
            f->Update();
            label = f->GetOutput();
        }
        return export_label(label);
    }

    if (kind == "watershed")
    {
        using WS = itk::WatershedImageFilter<ImageF>;
        auto f = WS::New();
        f->SetInput(input);
        f->SetThreshold(param_float(params, "threshold", 0.001f));
        f->SetLevel(param_float(params, "level", 0.3f));
        f->Update();
        // Watershed output is a label map of unsigned long; convert to uint8.
        using WSImage = itk::Image<itk::IdentifierType, 3>;
        py::array_t<std::uint8_t> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        itk::ImageRegionConstIterator<WSImage> it(f->GetOutput(), f->GetOutput()->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            const auto index = it.GetIndex();
            const std::size_t offset =
                (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
                 static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
                static_cast<std::size_t>(index[0]);
            dst[offset] = static_cast<std::uint8_t>(std::min<itk::IdentifierType>(it.Get(), 254));
        }
        return out;
    }

    if (kind == "fast_marching")
    {
        const ImageF::IndexType seed = seed_index(params);
        ImageF::Pointer arrival = fast_marching_init(input, params, seed);
        py::array_t<std::uint8_t> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        itk::ImageRegionConstIterator<ImageF> it(arrival, arrival->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            const auto index = it.GetIndex();
            const std::size_t offset =
                (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
                 static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
                static_cast<std::size_t>(index[0]);
            dst[offset] = it.Get() < 1.0e9f ? 1 : 0;
        }
        return out;
    }

    if (kind == "geodesic_active_contour" || kind == "shape_detection" ||
        kind == "threshold_segmentation" || kind == "laplacian_segmentation")
    {
        const ImageF::IndexType seed = seed_index(params);
        ImageF::Pointer init = fast_marching_init(input, params, seed);
        const int iterations = param_int(params, "iterations", 100);
        const float curvature = param_float(params, "curvature_scaling", 1.0f);
        const float propagation = param_float(params, "propagation_scaling", 1.0f);
        ImageF::Pointer level;

        if (kind == "geodesic_active_contour")
        {
            using LS = itk::GeodesicActiveContourLevelSetImageFilter<ImageF, ImageF>;
            auto f = LS::New();
            f->SetInput(init);
            f->SetFeatureImage(input);
            f->SetNumberOfIterations(iterations);
            f->SetPropagationScaling(propagation);
            f->SetCurvatureScaling(curvature);
            f->Update();
            level = f->GetOutput();
        }
        else if (kind == "shape_detection")
        {
            using LS = itk::ShapeDetectionLevelSetImageFilter<ImageF, ImageF>;
            auto f = LS::New();
            f->SetInput(init);
            f->SetFeatureImage(input);
            f->SetNumberOfIterations(iterations);
            f->SetPropagationScaling(propagation);
            f->SetCurvatureScaling(curvature);
            f->Update();
            level = f->GetOutput();
        }
        else if (kind == "threshold_segmentation")
        {
            using LS = itk::ThresholdSegmentationLevelSetImageFilter<ImageF, ImageF>;
            auto f = LS::New();
            f->SetInput(init);
            f->SetFeatureImage(input);
            f->SetNumberOfIterations(iterations);
            f->SetLowerThreshold(param_float(params, "lower_threshold", 0.0f));
            f->SetUpperThreshold(param_float(params, "upper_threshold", 255.0f));
            f->SetCurvatureScaling(curvature);
            f->SetPropagationScaling(propagation);
            f->Update();
            level = f->GetOutput();
        }
        else
        {
            using LS = itk::LaplacianSegmentationLevelSetImageFilter<ImageF, ImageF>;
            auto f = LS::New();
            f->SetInput(init);
            f->SetFeatureImage(input);
            f->SetNumberOfIterations(iterations);
            f->SetCurvatureScaling(curvature);
            f->SetPropagationScaling(propagation);
            f->Update();
            level = f->GetOutput();
        }

        py::array_t<std::uint8_t> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        itk::ImageRegionConstIterator<ImageF> it(level, level->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            const auto index = it.GetIndex();
            const std::size_t offset =
                (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
                 static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
                static_cast<std::size_t>(index[0]);
            dst[offset] = it.Get() <= 0.0f ? 1 : 0;
        }
        return out;
    }

    if (kind == "kmeans")
    {
        using KMeans = itk::ScalarImageKmeansImageFilter<ImageF>;
        auto f = KMeans::New();
        f->SetInput(input);
        const int num_clusters = param_int(params, "num_clusters", 3);
        // Initial means spread over the intensity range.
        itk::ImageRegionConstIterator<ImageF> it(input, input->GetLargestPossibleRegion());
        float lo = 0.0f, hi = 1.0f;
        bool first = true;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            const float v = it.Get();
            if (first) { lo = hi = v; first = false; }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        for (int c = 0; c < num_clusters; ++c)
        {
            const float mean = lo + (hi - lo) * (static_cast<float>(c) + 0.5f) / num_clusters;
            f->AddClassWithInitialMean(mean);
        }
        (void)params;  // Kd-tree KMeans converges internally
        f->Update();
        return export_label(f->GetOutput());
    }

    if (kind == "fcm")
    {
        // Native fuzzy c-means on intensity values.
        const int num_clusters = param_int(params, "num_clusters", 3);
        const float fuzziness = param_float(params, "fuzziness", 2.0f);
        const int max_iterations = param_int(params, "max_iterations", 100);
        const std::size_t count = static_cast<std::size_t>(ni * nj * nk);
        const float* src = values.data();

        float lo = src[0], hi = src[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            lo = std::min(lo, src[i]);
            hi = std::max(hi, src[i]);
        }
        std::vector<float> centers(static_cast<std::size_t>(num_clusters));
        for (int c = 0; c < num_clusters; ++c)
        {
            centers[static_cast<std::size_t>(c)] =
                lo + (hi - lo) * (static_cast<float>(c) + 0.5f) / static_cast<float>(num_clusters);
        }
        std::vector<float> u(static_cast<std::size_t>(num_clusters) * count);
        const float m = fuzziness;
        const float inv_m1 = 2.0f / (m - 1.0f);
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // Membership update.
            for (std::size_t i = 0; i < count; ++i)
            {
                float sum = 0.0f;
                for (int c = 0; c < num_clusters; ++c)
                {
                    const float d = std::max(std::abs(src[i] - centers[static_cast<std::size_t>(c)]), 1.0e-9f);
                    const float w = std::pow(1.0f / d, inv_m1);
                    u[static_cast<std::size_t>(c) * count + i] = w;
                    sum += w;
                }
                for (int c = 0; c < num_clusters; ++c)
                {
                    u[static_cast<std::size_t>(c) * count + i] /= sum;
                }
            }
            // Center update.
            std::vector<float> new_centers(static_cast<std::size_t>(num_clusters), 0.0f);
            std::vector<float> weight_sum(static_cast<std::size_t>(num_clusters), 0.0f);
            for (int c = 0; c < num_clusters; ++c)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const float w = std::pow(u[static_cast<std::size_t>(c) * count + i], m);
                    new_centers[static_cast<std::size_t>(c)] += w * src[i];
                    weight_sum[static_cast<std::size_t>(c)] += w;
                }
                new_centers[static_cast<std::size_t>(c)] /= std::max(weight_sum[static_cast<std::size_t>(c)], 1.0e-12f);
            }
            float delta = 0.0f;
            for (int c = 0; c < num_clusters; ++c)
            {
                delta = std::max(delta, std::abs(new_centers[static_cast<std::size_t>(c)] - centers[static_cast<std::size_t>(c)]));
            }
            centers = new_centers;
            if (delta < 1.0e-4f)
            {
                break;
            }
        }
        py::array_t<std::uint8_t> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        for (std::size_t i = 0; i < count; ++i)
        {
            int best = 0;
            float best_u = -1.0f;
            for (int c = 0; c < num_clusters; ++c)
            {
                const float val = u[static_cast<std::size_t>(c) * count + i];
                if (val > best_u) { best_u = val; best = c; }
            }
            dst[i] = static_cast<std::uint8_t>(best + 1);
        }
        return out;
    }

    if (kind == "island_remove")
    {
        // Connected components; keep the largest.
        using LabelImage = itk::Image<std::uint32_t, 3>;
        using CC = itk::ConnectedComponentImageFilter<ImageU8, LabelImage>;
        using Relabel = itk::RelabelComponentImageFilter<LabelImage, LabelImage>;
        using Bin = itk::BinaryThresholdImageFilter<ImageF, ImageU8>;
        auto bin = Bin::New();
        bin->SetInput(input);
        bin->SetLowerThreshold(0.5f);
        bin->SetUpperThreshold(1.0e30f);
        bin->SetInsideValue(1);
        bin->SetOutsideValue(0);
        bin->Update();
        auto cc = CC::New();
        cc->SetInput(bin->GetOutput());
        cc->Update();
        auto relabel = Relabel::New();
        relabel->SetInput(cc->GetOutput());
        relabel->Update();
        LabelImage::Pointer labels = relabel->GetOutput();
        itk::ImageRegionIterator<LabelImage> it(labels, labels->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            it.Set(it.Get() > 1 ? 0 : it.Get());
        }
        // Downcast to uint8.
        py::array_t<std::uint8_t> out8({nk, nj, ni});
        auto* dst8 = out8.mutable_data();
        itk::ImageRegionConstIterator<LabelImage> lit(labels, labels->GetLargestPossibleRegion());
        for (lit.GoToBegin(); !lit.IsAtEnd(); ++lit)
        {
            const auto index = lit.GetIndex();
            const std::size_t offset =
                (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(nj) +
                 static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(ni) +
                static_cast<std::size_t>(index[0]);
            dst8[offset] = static_cast<std::uint8_t>(std::min<std::uint32_t>(lit.Get(), 255));
        }
        return out8;
    }

    throw std::invalid_argument("segment_3d: unknown kind '" + kind + "'");
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

py::array_t<float> register_pair(
    py::array_t<float, py::array::c_style | py::array::forcecast> fixed,
    py::array_t<float, py::array::c_style | py::array::forcecast> moving,
    const std::string& kind, const py::dict& params)
{
    if (fixed.ndim() != 3 || moving.ndim() != 3)
    {
        throw std::invalid_argument("register_pair expects two 3-D arrays");
    }
    const py::ssize_t nk = fixed.shape(0), nj = fixed.shape(1), ni = fixed.shape(2);
    ImageF::Pointer fixed_img = import_image(fixed.data(), nk, nj, ni);
    ImageF::Pointer moving_img = import_image(moving.data(), nk, nj, ni);

    if (kind == "demons" || kind == "diffeomorphic_demons" || kind == "fast_symmetric_demons")
    {
        using FieldImage = itk::Image<itk::Vector<float, 3>, 3>;
        using Warper = itk::WarpImageFilter<ImageF, ImageF, FieldImage>;
        FieldImage::Pointer field;
        if (kind == "demons")
        {
            using Demons = itk::DemonsRegistrationFilter<ImageF, ImageF, FieldImage>;
            auto f = Demons::New();
            f->SetFixedImage(fixed_img);
            f->SetMovingImage(moving_img);
            f->SetNumberOfIterations(param_int(params, "iterations", 20));
            f->SetStandardDeviations(param_float(params, "standard_deviation", 1.0f));
            f->Update();
            field = f->GetOutput();
        }
        else if (kind == "diffeomorphic_demons")
        {
            using Demons = itk::DiffeomorphicDemonsRegistrationFilter<ImageF, ImageF, FieldImage>;
            auto f = Demons::New();
            f->SetFixedImage(fixed_img);
            f->SetMovingImage(moving_img);
            f->SetNumberOfIterations(param_int(params, "iterations", 20));
            f->SetStandardDeviations(param_float(params, "standard_deviation", 1.0f));
            f->Update();
            field = f->GetOutput();
        }
        else
        {
            using Demons = itk::FastSymmetricForcesDemonsRegistrationFilter<ImageF, ImageF, FieldImage>;
            auto f = Demons::New();
            f->SetFixedImage(fixed_img);
            f->SetMovingImage(moving_img);
            f->SetNumberOfIterations(param_int(params, "iterations", 20));
            f->SetStandardDeviations(param_float(params, "standard_deviation", 1.0f));
            f->Update();
            field = f->GetOutput();
        }

        auto warper = Warper::New();
        warper->SetInput(moving_img);
        warper->SetDisplacementField(field);
        warper->SetOutputSpacing(fixed_img->GetSpacing());
        warper->SetOutputOrigin(fixed_img->GetOrigin());
        warper->SetOutputDirection(fixed_img->GetDirection());
        warper->Update();
        return export_float(warper->GetOutput());
    }

    // ITKv4 registration for rigid / affine / bspline.
    using Metric = itk::MattesMutualInformationImageToImageMetricv4<ImageF, ImageF>;
    using Optimizer = itk::RegularStepGradientDescentOptimizerv4<double>;
    using Registration = itk::ImageRegistrationMethodv4<ImageF, ImageF>;

    typename itk::Transform<double, 3, 3>::Pointer transform;
    if (kind == "rigid")
    {
        transform = itk::VersorRigid3DTransform<double>::New();
    }
    else if (kind == "affine")
    {
        transform = itk::AffineTransform<double, 3>::New();
    }
    else if (kind == "bspline")
    {
        using BSp = itk::BSplineTransform<double, 3>;
        auto bsp = BSp::New();
        const std::vector<int> mesh = py::cast<std::vector<int>>(params["mesh_size"]);
        bsp->SetTransformDomainOrigin(fixed_img->GetOrigin());
        typename BSp::PhysicalDimensionsType dims;
        const auto size = fixed_img->GetLargestPossibleRegion().GetSize();
        const auto spacing = fixed_img->GetSpacing();
        for (int d = 0; d < 3; ++d)
        {
            dims[d] = spacing[d] * static_cast<double>(size[d] - 1);
        }
        bsp->SetTransformDomainPhysicalDimensions(dims);
        typename BSp::MeshSizeType mesh_size;
        mesh_size[0] = mesh[0];
        mesh_size[1] = mesh[1];
        mesh_size[2] = mesh[2];
        bsp->SetTransformDomainMeshSize(mesh_size);
        bsp->SetTransformDomainDirection(fixed_img->GetDirection());
        transform = bsp;
    }
    else
    {
        throw std::invalid_argument("register_pair: unknown kind '" + kind + "'");
    }

    auto metric = Metric::New();
    metric->SetNumberOfHistogramBins(param_int(params, "histogram_bins", 32));
    auto optimizer = Optimizer::New();
    optimizer->SetLearningRate(param_float(params, "learning_rate", 0.1f));
    optimizer->SetNumberOfIterations(param_int(params, "iterations", 100));
    optimizer->SetMinimumStepLength(1.0e-4);
    optimizer->SetRelaxationFactor(0.5);

    auto registration = Registration::New();
    registration->SetFixedImage(fixed_img);
    registration->SetMovingImage(moving_img);
    registration->SetInitialTransform(transform);
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    registration->SetNumberOfLevels(1);
    registration->Update();

    auto resampler = itk::ResampleImageFilter<ImageF, ImageF>::New();
    resampler->SetInput(moving_img);
    resampler->SetTransform(registration->GetTransform());
    resampler->SetSize(fixed_img->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputSpacing(fixed_img->GetSpacing());
    resampler->SetOutputOrigin(fixed_img->GetOrigin());
    resampler->SetOutputDirection(fixed_img->GetDirection());
    resampler->Update();
    return export_float(resampler->GetOutput());
}

} // namespace native_ops
} // namespace medaxis
