/*
 * Native ITK-backed operations: filtering, morphology, arithmetic,
 * enhancement, geometry. All algorithms execute in C++/ITK; numpy buffers
 * are imported zero-copy and results are exported as fresh arrays.
 */

#include "native_ops.h"
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <itkAdaptiveHistogramEqualizationImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkBilateralImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryFillholeImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCannyEdgeDetectionImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkFlipImageFilter.h>
#include <itkGradientAnisotropicDiffusionImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkGrayscaleErodeImageFilter.h>
#include <itkGrayscaleMorphologicalClosingImageFilter.h>
#include <itkGrayscaleMorphologicalOpeningImageFilter.h>
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImportImageFilter.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkInvertIntensityImageFilter.h>
#include <itkLaplacianRecursiveGaussianImageFilter.h>
#include <itkLog10ImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkMeanImageFilter.h>
#include <itkMedianImageFilter.h>
#include <itkMorphologicalGradientImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkPatchBasedDenoisingBaseImageFilter.h>
#include <itkPatchBasedDenoisingImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkPermuteAxesImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include <itkSobelEdgeDetectionImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkWhiteTopHatImageFilter.h>
#include <itkBlackTopHatImageFilter.h>
#include <itkZeroCrossingImageFilter.h>

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

inline bool param_bool(const py::dict& params, const char* key, bool def)
{
    if (!params.contains(key)) return def;
    return py::cast<bool>(params[key]);
}

/// Zero-copy import of a (k, j, i) float32 numpy array into an ITK image.
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

/// Export an ITK float image to a fresh (k, j, i) numpy array.
py::array_t<float> export_image(ImageF::Pointer image)
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

/// Export an ITK uint8 image to a fresh (k, j, i) numpy array.
py::array_t<std::uint8_t> export_image_u8(ImageU8::Pointer image)
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

/// Binary structuring element (ball / box / cross).
template <typename TElem>
TElem make_kernel(const py::dict& params, int radius)
{
    TElem kernel;
    kernel.SetRadius(radius);
    kernel.CreateStructuringElement();
    return kernel;
}

} // namespace

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

py::array_t<float> filter_3d(py::array_t<float, py::array::c_style | py::array::forcecast> values,
                             const std::string& kind, const py::dict& params)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("filter_3d expects a 3-D array (z, y, x)");
    }
    const py::ssize_t nk = values.shape(0), nj = values.shape(1), ni = values.shape(2);
    ImageF::Pointer input = import_image(values.data(), nk, nj, ni);

    if (kind == "cast")
    {
        return py::array_t<float>({nk, nj, ni}, values.data());
    }
    if (kind == "sharpen")
    {
        const float amount = param_float(params, "amount", 1.0f);
        auto lap = itk::LaplacianRecursiveGaussianImageFilter<ImageF, ImageF>::New();
        lap->SetInput(input);
        lap->SetSigma(1.0);
        lap->Update();
        auto add = itk::AddImageFilter<ImageF, ImageF, ImageF>::New();
        add->SetInput1(input);
        add->SetInput2(lap->GetOutput());
        add->Update();
        auto out = add->GetOutput();
        if (std::abs(amount - 1.0f) > 1e-6f)
        {
            itk::ImageRegionIterator<ImageF> it(out, out->GetLargestPossibleRegion());
            for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                it.Set(it.Get() * amount + input->GetPixel(it.GetIndex()) * (1.0f - amount));
            }
        }
        return export_image(out);
    }

    using Smooth = itk::SmoothingRecursiveGaussianImageFilter<ImageF, ImageF>;
    using Median = itk::MedianImageFilter<ImageF, ImageF>;
    using Bilateral = itk::BilateralImageFilter<ImageF, ImageF>;
    using GradMag = itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageF, ImageF>;
    using Laplacian = itk::LaplacianRecursiveGaussianImageFilter<ImageF, ImageF>;
    using AnisoDiff = itk::GradientAnisotropicDiffusionImageFilter<ImageF, ImageF>;
    using Mean = itk::MeanImageFilter<ImageF, ImageF>;
    using DiscGauss = itk::DiscreteGaussianImageFilter<ImageF, ImageF>;
    using Sobel = itk::SobelEdgeDetectionImageFilter<ImageF, ImageF>;
    using Canny = itk::CannyEdgeDetectionImageFilter<ImageF, ImageF>;
    using ZeroX = itk::ZeroCrossingImageFilter<ImageF, ImageF>;
    using Normalize = itk::NormalizeImageFilter<ImageF, ImageF>;
    using Rescale = itk::RescaleIntensityImageFilter<ImageF, ImageF>;

    if (kind == "gaussian")
    {
        auto f = Smooth::New();
        f->SetInput(input);
        f->SetSigma(param_float(params, "sigma", 1.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "median")
    {
        auto f = Median::New();
        f->SetInput(input);
        Median::RadiusType radius;
        radius.Fill(param_int(params, "radius", 2));
        f->SetRadius(radius);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "bilateral")
    {
        auto f = Bilateral::New();
        f->SetInput(input);
        f->SetDomainSigma(param_float(params, "domain_sigma", 2.0f));
        f->SetRangeSigma(param_float(params, "range_sigma", 50.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "gradient_magnitude")
    {
        auto f = GradMag::New();
        f->SetInput(input);
        f->SetSigma(param_float(params, "sigma", 1.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "laplacian")
    {
        auto f = Laplacian::New();
        f->SetInput(input);
        f->SetSigma(param_float(params, "sigma", 1.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "anisotropic_diffusion")
    {
        auto f = AnisoDiff::New();
        f->SetInput(input);
        f->SetNumberOfIterations(param_int(params, "iterations", 5));
        f->SetTimeStep(param_float(params, "time_step", 0.0625f));
        f->SetConductanceParameter(param_float(params, "conductance", 3.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "mean")
    {
        auto f = Mean::New();
        f->SetInput(input);
        Mean::RadiusType radius;
        radius.Fill(param_int(params, "radius", 2));
        f->SetRadius(radius);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "discrete_gaussian")
    {
        auto f = DiscGauss::New();
        f->SetInput(input);
        f->SetVariance(param_float(params, "variance", 1.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "sobel")
    {
        auto f = Sobel::New();
        f->SetInput(input);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "canny")
    {
        auto f = Canny::New();
        f->SetInput(input);
        f->SetLowerThreshold(param_float(params, "lower_threshold", 20.0f));
        f->SetUpperThreshold(param_float(params, "upper_threshold", 60.0f));
        f->SetVariance(param_float(params, "variance", 2.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "zero_crossing")
    {
        auto f = ZeroX::New();
        f->SetInput(input);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "normalize")
    {
        auto f = Normalize::New();
        f->SetInput(input);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "rescale")
    {
        auto f = Rescale::New();
        f->SetInput(input);
        f->SetOutputMinimum(param_float(params, "output_minimum", 0.0f));
        f->SetOutputMaximum(param_float(params, "output_maximum", 255.0f));
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "nl_means")
    {
        // Patch-based denoising (non-local means family) with default params.
        using PatchDenoise = itk::PatchBasedDenoisingImageFilter<ImageF, ImageF>;
        auto f = PatchDenoise::New();
        f->SetInput(input);
        f->SetKernelBandwidthEstimation(true);
        f->SetNoiseModel(
            itk::PatchBasedDenoisingBaseImageFilterEnums::NoiseModel::GAUSSIAN);
        f->SetNumberOfIterations(param_int(params, "iterations", 1));
        f->Update();
        return export_image(f->GetOutput());
    }

    throw std::invalid_argument("filter_3d: unknown kind '" + kind + "'");
}

// ---------------------------------------------------------------------------
// Morphology
// ---------------------------------------------------------------------------

py::array morphology_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind, const py::dict& params)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("morphology_3d expects a 3-D array (z, y, x)");
    }
    const py::ssize_t nk = values.shape(0), nj = values.shape(1), ni = values.shape(2);
    const int radius = param_int(params, "radius", 1);

    using BallF = itk::BinaryBallStructuringElement<float, 3>;

    if (kind == "connected_components")
    {
        using LabelImage = itk::Image<std::uint32_t, 3>;
        using CC = itk::ConnectedComponentImageFilter<ImageU8, LabelImage>;
        using Relabel = itk::RelabelComponentImageFilter<LabelImage, LabelImage>;
        auto bin = itk::BinaryThresholdImageFilter<ImageF, ImageU8>::New();
        bin->SetInput(import_image(values.data(), nk, nj, ni));
        bin->SetLowerThreshold(0.5f);
        bin->SetUpperThreshold(1.0e30f);
        bin->SetInsideValue(1);
        bin->SetOutsideValue(0);
        bin->Update();
        auto cc = CC::New();
        cc->SetInput(bin->GetOutput());
        cc->SetFullyConnected(param_bool(params, "fully_connected", false));
        cc->Update();
        auto relabel = Relabel::New();
        relabel->SetInput(cc->GetOutput());
        relabel->Update();
        LabelImage::Pointer labels = relabel->GetOutput();
        if (param_bool(params, "keep_largest", false))
        {
            // Keep only the largest component (relabel orders by size descending).
            itk::ImageRegionIterator<LabelImage> it(labels, labels->GetLargestPossibleRegion());
            for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                it.Set(it.Get() > 1 ? 0 : it.Get());
            }
        }
        // Downcast to uint8 (labels beyond 255 are clamped).
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

    // Binary mask (foreground = value > 0.5).
    auto bin = itk::BinaryThresholdImageFilter<ImageF, ImageU8>::New();
    bin->SetInput(import_image(values.data(), nk, nj, ni));
    bin->SetLowerThreshold(0.5f);
    bin->SetUpperThreshold(1.0e30f);
    bin->SetInsideValue(1);
    bin->SetOutsideValue(0);
    bin->Update();

    if (kind == "hole_filling")
    {
        auto f = itk::BinaryFillholeImageFilter<ImageU8>::New();
        f->SetInput(bin->GetOutput());
        f->SetFullyConnected(param_bool(params, "fully_connected", false));
        f->Update();
        return export_image_u8(f->GetOutput());
    }

    using BinErode = itk::BinaryErodeImageFilter<ImageU8, ImageU8, BallF>;
    using BinDilate = itk::BinaryDilateImageFilter<ImageU8, ImageU8, BallF>;
    using GrayErode = itk::GrayscaleErodeImageFilter<ImageF, ImageF, BallF>;
    using GrayDilate = itk::GrayscaleDilateImageFilter<ImageF, ImageF, BallF>;
    using GrayOpen = itk::GrayscaleMorphologicalOpeningImageFilter<ImageF, ImageF, BallF>;
    using GrayClose = itk::GrayscaleMorphologicalClosingImageFilter<ImageF, ImageF, BallF>;
    using TopHat = itk::WhiteTopHatImageFilter<ImageF, ImageF, BallF>;
    using BotHat = itk::BlackTopHatImageFilter<ImageF, ImageF, BallF>;
    using MorphGrad = itk::MorphologicalGradientImageFilter<ImageF, ImageF, BallF>;

    ImageF::Pointer input = import_image(values.data(), nk, nj, ni);

    // Binary operations produce a uint8 label volume.
    if (kind == "binary_erode")
    {
        auto f = BinErode::New();
        f->SetInput(bin->GetOutput());
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        return export_image_u8(f->GetOutput());
    }
    if (kind == "binary_dilate")
    {
        auto f = BinDilate::New();
        f->SetInput(bin->GetOutput());
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        return export_image_u8(f->GetOutput());
    }
    if (kind == "binary_open")
    {
        auto e = BinErode::New();
        e->SetInput(bin->GetOutput());
        e->SetKernel(make_kernel<BallF>(params, radius));
        auto d = BinDilate::New();
        d->SetInput(e->GetOutput());
        d->SetKernel(make_kernel<BallF>(params, radius));
        d->Update();
        return export_image_u8(d->GetOutput());
    }
    if (kind == "binary_close")
    {
        auto d = BinDilate::New();
        d->SetInput(bin->GetOutput());
        d->SetKernel(make_kernel<BallF>(params, radius));
        auto e = BinErode::New();
        e->SetInput(d->GetOutput());
        e->SetKernel(make_kernel<BallF>(params, radius));
        e->Update();
        return export_image_u8(e->GetOutput());
    }

    // Grayscale operations preserve float output.
    ImageF::Pointer result;
    if (kind == "grayscale_erode")
    {
        auto f = GrayErode::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_dilate")
    {
        auto f = GrayDilate::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_open")
    {
        auto f = GrayOpen::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_close")
    {
        auto f = GrayClose::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_tophat")
    {
        auto f = TopHat::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_bottomhat")
    {
        auto f = BotHat::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else if (kind == "grayscale_gradient")
    {
        auto f = MorphGrad::New();
        f->SetInput(input);
        f->SetKernel(make_kernel<BallF>(params, radius));
        f->Update();
        result = f->GetOutput();
    }
    else
    {
        throw std::invalid_argument("morphology_3d: unknown kind '" + kind + "'");
    }
    // Morphology outputs are returned as float arrays (grayscale path).
    return export_image(result);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

py::array_t<float> arith_3d(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                            const py::object& b, const std::string& kind, const py::dict& params)
{
    if (a.ndim() != 3)
    {
        throw std::invalid_argument("arith_3d expects 3-D arrays");
    }
    const py::ssize_t nk = a.shape(0), nj = a.shape(1), ni = a.shape(2);
    ImageF::Pointer ia = import_image(a.data(), nk, nj, ni);
    ImageF::Pointer ib;
    const bool b_is_array = py::isinstance<py::array>(b);
    if (b_is_array)
    {
        auto b_arr = b.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        if (b_arr.ndim() != 3 || b_arr.shape(0) != nk || b_arr.shape(1) != nj || b_arr.shape(2) != ni)
        {
            throw std::invalid_argument("arith_3d: operand array must match dimensions");
        }
        ib = import_image(b_arr.data(), nk, nj, ni);
    }
    const float constant = b_is_array ? 0.0f : static_cast<float>(py::cast<double>(b));

    if (kind == "invert")
    {
        auto f = itk::InvertIntensityImageFilter<ImageF>::New();
        f->SetInput(ia);
        const float maximum = param_float(params, "maximum", -1.0f);
        if (maximum > 0.0f)
        {
            f->SetMaximum(maximum);
        }
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "mask")
    {
        // Direct loop: keep voxels where the mask != 0, else outside_value.
        const std::size_t count = static_cast<std::size_t>(ni * nj * nk);
        const float outside = param_float(params, "outside_value", 0.0f);
        py::array_t<float> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        const float* src = a.data();
        if (b_is_array)
        {
            auto b_arr = b.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            const float* mask_ptr = b_arr.data();
            for (std::size_t i = 0; i < count; ++i)
            {
                dst[i] = mask_ptr[i] != 0.0f ? src[i] : outside;
            }
        }
        else
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                dst[i] = src[i] > 0.5f ? src[i] : outside;
            }
        }
        return out;
    }

    using Add = itk::AddImageFilter<ImageF, ImageF, ImageF>;
    using Sub = itk::SubtractImageFilter<ImageF, ImageF, ImageF>;
    using Mul = itk::MultiplyImageFilter<ImageF, ImageF, ImageF>;
    using Div = itk::DivideImageFilter<ImageF, ImageF, ImageF>;

    if (kind == "add")
    {
        auto f = Add::New();
        f->SetInput1(ia);
        if (b_is_array) { f->SetInput2(ib); } else { f->SetConstant2(constant); }
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "subtract")
    {
        auto f = Sub::New();
        f->SetInput1(ia);
        if (b_is_array) { f->SetInput2(ib); } else { f->SetConstant2(constant); }
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "multiply")
    {
        auto f = Mul::New();
        f->SetInput1(ia);
        if (b_is_array) { f->SetInput2(ib); } else { f->SetConstant2(constant); }
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "divide")
    {
        auto f = Div::New();
        f->SetInput1(ia);
        if (b_is_array) { f->SetInput2(ib); } else { f->SetConstant2(constant); }
        f->Update();
        return export_image(f->GetOutput());
    }
    throw std::invalid_argument("arith_3d: unknown kind '" + kind + "'");
}

// ---------------------------------------------------------------------------
// Enhancement
// ---------------------------------------------------------------------------

py::array_t<float> enhance_3d(py::array_t<float, py::array::c_style | py::array::forcecast> values,
                              const std::string& kind, const py::dict& params)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("enhance_3d expects a 3-D array");
    }
    const py::ssize_t nk = values.shape(0), nj = values.shape(1), ni = values.shape(2);
    ImageF::Pointer input = import_image(values.data(), nk, nj, ni);

    if (kind == "window_level")
    {
        auto f = itk::IntensityWindowingImageFilter<ImageF, ImageF>::New();
        f->SetInput(input);
        const float window = param_float(params, "window", 400.0f);
        const float level = param_float(params, "level", 40.0f);
        f->SetWindowMinimum(level - window / 2.0f);
        f->SetWindowMaximum(level + window / 2.0f);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "clahe")
    {
        auto f = itk::AdaptiveHistogramEqualizationImageFilter<ImageF>::New();
        f->SetInput(input);
        f->SetAlpha(param_float(params, "alpha", 0.0f));
        f->SetBeta(param_float(params, "beta", 1.0f));
        const int kernel = param_int(params, "kernel_size", 8);
        itk::AdaptiveHistogramEqualizationImageFilter<ImageF>::RadiusType radius;
        radius.Fill(kernel);
        f->SetRadius(radius);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "histogram_equalization")
    {
        auto f = itk::AdaptiveHistogramEqualizationImageFilter<ImageF>::New();
        f->SetInput(input);
        f->SetAlpha(0.0f);
        f->SetBeta(1.0f);
        itk::AdaptiveHistogramEqualizationImageFilter<ImageF>::RadiusType radius;
        radius.Fill(0); // global histogram equalization
        f->SetRadius(radius);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "log_transform")
    {
        auto f = itk::Log10ImageFilter<ImageF, ImageF>::New();
        f->SetInput(input);
        f->Update();
        auto out = f->GetOutput();
        const float shift = param_float(params, "shift", 0.0f);
        const float scale = param_float(params, "scale", 1.0f);
        if (std::abs(shift) > 1e-6f || std::abs(scale - 1.0f) > 1e-6f)
        {
            itk::ImageRegionIterator<ImageF> it(out, out->GetLargestPossibleRegion());
            itk::ImageRegionConstIterator<ImageF> itin(input, input->GetLargestPossibleRegion());
            for (itin.GoToBegin(), it.GoToBegin(); !itin.IsAtEnd(); ++itin, ++it)
            {
                const double v = std::max(itin.Get() + static_cast<double>(shift), 0.0);
                it.Set(static_cast<float>(std::log1p(v) * static_cast<double>(scale)));
            }
        }
        return export_image(out);
    }
    if (kind == "gamma")
    {
        const float gamma = param_float(params, "gamma", 1.0f);
        py::array_t<float> out({nk, nj, ni});
        auto* dst = out.mutable_data();
        const float* src = values.data();
        const std::size_t count = static_cast<std::size_t>(ni * nj * nk);
        float lo = src[0], hi = src[0];
        for (std::size_t i = 1; i < count; ++i)
        {
            lo = std::min(lo, src[i]);
            hi = std::max(hi, src[i]);
        }
        const float range = std::max(hi - lo, 1.0e-12f);
        for (std::size_t i = 0; i < count; ++i)
        {
            const float norm = std::clamp((src[i] - lo) / range, 0.0f, 1.0f);
            dst[i] = std::pow(norm, gamma);
        }
        return out;
    }
    throw std::invalid_argument("enhance_3d: unknown kind '" + kind + "'");
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

py::array_t<float> geometry_3d(py::array_t<float, py::array::c_style | py::array::forcecast> values,
                               const std::string& kind, const py::dict& params)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("geometry_3d expects a 3-D array");
    }
    const py::ssize_t nk = values.shape(0), nj = values.shape(1), ni = values.shape(2);
    ImageF::Pointer input = import_image(values.data(), nk, nj, ni);

    if (kind == "flip")
    {
        auto f = itk::FlipImageFilter<ImageF>::New();
        f->SetInput(input);
        itk::FlipImageFilter<ImageF>::FlipAxesArrayType axes;
        axes[0] = param_bool(params, "flip_x", false) ? 1 : 0;
        axes[1] = param_bool(params, "flip_y", false) ? 1 : 0;
        axes[2] = param_bool(params, "flip_z", false) ? 1 : 0;
        f->SetFlipAxes(axes);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "permute")
    {
        auto f = itk::PermuteAxesImageFilter<ImageF>::New();
        f->SetInput(input);
        itk::PermuteAxesImageFilter<ImageF>::PermuteOrderArrayType order;
        const std::vector<int> ord = py::cast<std::vector<int>>(params["order"]);
        if (ord.size() != 3)
        {
            throw std::invalid_argument("permute: order must have 3 entries");
        }
        for (int d = 0; d < 3; ++d)
        {
            order[d] = static_cast<unsigned int>(ord[d]);
        }
        f->SetOrder(order);
        f->Update();
        return export_image(f->GetOutput());
    }
    if (kind == "crop")
    {
        const std::vector<int> bounds = py::cast<std::vector<int>>(params["bounds"]);
        if (bounds.size() != 6)
        {
            throw std::invalid_argument("crop: bounds must be [i0,i1,j0,j1,k0,k1]");
        }
        auto clamp = [](int v, py::ssize_t lo, py::ssize_t hi) {
            return static_cast<py::ssize_t>(std::max(lo, std::min<py::ssize_t>(v, hi)));
        };
        const py::ssize_t i0 = clamp(bounds[0], 0, ni), i1 = clamp(bounds[1], 0, ni);
        const py::ssize_t j0 = clamp(bounds[2], 0, nj), j1 = clamp(bounds[3], 0, nj);
        const py::ssize_t k0 = clamp(bounds[4], 0, nk), k1 = clamp(bounds[5], 0, nk);
        const py::ssize_t oni = std::max<py::ssize_t>(i1 - i0, 1);
        const py::ssize_t onj = std::max<py::ssize_t>(j1 - j0, 1);
        const py::ssize_t onk = std::max<py::ssize_t>(k1 - k0, 1);
        py::array_t<float> out({onk, onj, oni});
        auto* dst = out.mutable_data();
        const float* src = values.data();
        for (py::ssize_t k = 0; k < onk; ++k)
        {
            for (py::ssize_t j = 0; j < onj; ++j)
            {
                const float* row = src + ((k0 + k) * nj + (j0 + j)) * ni + i0;
                std::memcpy(dst + (k * onj + j) * oni, row, sizeof(float) * static_cast<std::size_t>(oni));
            }
        }
        return out;
    }
    if (kind == "pad")
    {
        const std::vector<int> pad = py::cast<std::vector<int>>(params["pad"]);
        if (pad.size() != 3)
        {
            throw std::invalid_argument("pad: pad must have 3 entries");
        }
        const float value = param_float(params, "value", 0.0f);
        const py::ssize_t pi = pad[0], pj = pad[1], pk = pad[2];
        const py::ssize_t oni = ni + 2 * pi, onj = nj + 2 * pj, onk = nk + 2 * pk;
        py::array_t<float> out({onk, onj, oni});
        auto* dst = out.mutable_data();
        const std::size_t total = static_cast<std::size_t>(onk * onj * oni);
        std::fill(dst, dst + total, value);
        const float* src = values.data();
        for (py::ssize_t k = 0; k < nk; ++k)
        {
            for (py::ssize_t j = 0; j < nj; ++j)
            {
                const float* row = src + (k * nj + j) * ni;
                std::memcpy(dst + ((k + pk) * onj + (j + pj)) * oni + pi,
                            row, sizeof(float) * static_cast<std::size_t>(ni));
            }
        }
        return out;
    }
    if (kind == "resample")
    {
        const std::vector<double> spacing = py::cast<std::vector<double>>(params["spacing"]);
        if (spacing.size() != 3)
        {
            throw std::invalid_argument("resample: spacing must have 3 entries");
        }
        auto f = itk::ResampleImageFilter<ImageF, ImageF>::New();
        f->SetInput(input);
        ImageF::SpacingType out_spacing;
        out_spacing[0] = spacing[0];
        out_spacing[1] = spacing[1];
        out_spacing[2] = spacing[2];
        f->SetOutputSpacing(out_spacing);
        ImageF::SizeType size;
        const auto in_spacing = input->GetSpacing();
        const auto in_size = input->GetLargestPossibleRegion().GetSize();
        for (int d = 0; d < 3; ++d)
        {
            size[d] = static_cast<unsigned int>(
                std::max(1LL, static_cast<long long>(std::llround(
                    static_cast<double>(in_size[d]) * in_spacing[d] / spacing[d]))));
        }
        f->SetSize(size);
        f->SetOutputOrigin(input->GetOrigin());
        f->SetOutputDirection(input->GetDirection());
        f->Update();
        return export_image(f->GetOutput());
    }
    throw std::invalid_argument("geometry_3d: unknown kind '" + kind + "'");
}

} // namespace native_ops
} // namespace medaxis
