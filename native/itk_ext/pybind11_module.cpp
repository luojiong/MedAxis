#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImportImageFilter.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "segmentation/medAxisThresholdFilter.h"

namespace py = pybind11;

namespace
{

using InputImage = itk::Image<float, 3>;
using OutputImage = itk::Image<std::uint8_t, 3>;
using ImportFilter = itk::ImportImageFilter<float, 3>;
using ThresholdFilter = medaxis::ThresholdFilter<InputImage, OutputImage>;

medaxis::ThresholdMethod parse_method(const std::string& method)
{
    if (method == "otsu") return medaxis::ThresholdMethod::Otsu;
    if (method == "adaptive") return medaxis::ThresholdMethod::Adaptive;
    if (method == "isodata") return medaxis::ThresholdMethod::IsoData;
    if (method == "max_entropy") return medaxis::ThresholdMethod::MaxEntropy;
    if (method == "moments") return medaxis::ThresholdMethod::Moments;
    if (method == "renyi_entropy") return medaxis::ThresholdMethod::RenyiEntropy;
    if (method == "huang") return medaxis::ThresholdMethod::Huang;
    if (method == "intermodes") return medaxis::ThresholdMethod::Intermodes;
    if (method == "li") return medaxis::ThresholdMethod::Li;
    if (method == "manual") return medaxis::ThresholdMethod::Manual;
    throw std::invalid_argument("unknown threshold method: " + method);
}

py::array_t<std::uint8_t> threshold_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& method,
    double lower,
    double upper,
    unsigned int histogram_bins,
    unsigned int adaptive_radius)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("threshold_3d expects a 3-D array with shape (z, y, x)");
    }

    const auto z = static_cast<py::ssize_t>(values.shape(0));
    const auto y = static_cast<py::ssize_t>(values.shape(1));
    const auto x = static_cast<py::ssize_t>(values.shape(2));
    if (x == 0 || y == 0 || z == 0)
    {
        throw std::invalid_argument("threshold_3d expects a non-empty array");
    }

    const std::size_t voxel_count = static_cast<std::size_t>(x * y * z);
    std::vector<float> input_buffer(values.data(), values.data() + voxel_count);

    auto importer = ImportFilter::New();
    InputImage::SizeType size;
    size[0] = static_cast<InputImage::SizeType::SizeValueType>(x);
    size[1] = static_cast<InputImage::SizeType::SizeValueType>(y);
    size[2] = static_cast<InputImage::SizeType::SizeValueType>(z);
    InputImage::RegionType region;
    region.SetSize(size);
    importer->SetRegion(region);
    importer->SetImportPointer(input_buffer.data(), input_buffer.size(), false);
    importer->Update();

    auto filter = ThresholdFilter::New();
    filter->SetInput(importer->GetOutput());
    filter->SetMethod(parse_method(method));
    filter->SetLowerThreshold(lower);
    filter->SetUpperThreshold(upper);
    filter->SetNumberOfHistogramBins(histogram_bins);
    filter->SetAdaptiveRadius(adaptive_radius);
    filter->Update();

    py::array_t<std::uint8_t> output({z, y, x});
    auto output_buffer = output.mutable_data();
    itk::ImageRegionConstIterator<OutputImage> iterator(filter->GetOutput(), region);
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
        const auto index = iterator.GetIndex();
        const std::size_t offset =
            (static_cast<std::size_t>(index[2]) * static_cast<std::size_t>(y) +
             static_cast<std::size_t>(index[1])) * static_cast<std::size_t>(x) +
            static_cast<std::size_t>(index[0]);
        output_buffer[offset] = iterator.Get();
    }
    return output;
}

} // namespace

PYBIND11_MODULE(medaxis_itk, module)
{
    module.doc() = "MedAxis ITK-backed image processing extensions.";
    module.def(
        "threshold_3d",
        &threshold_3d,
        py::arg("values"),
        py::arg("method") = "otsu",
        py::arg("lower") = 0.0,
        py::arg("upper") = 0.0,
        py::arg("histogram_bins") = 128,
        py::arg("adaptive_radius") = 5,
        "Threshold a 3-D float array with the MedAxis ITK threshold filter.");
}
