#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace medaxis
{
namespace radiomics_ext
{

/// Texture radiomics: glcm(24) + glrlm(16) + glszm(16) + gldm(14) + ngtdm(5).
/// `values` is a 3-D float array (k, j, i); intensities are quantized to
/// `levels` bins.  Returns a {feature_name: value} dictionary.
py::dict texture_features(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    int levels,
    const std::vector<std::string>& families);

} // namespace radiomics_ext
} // namespace medaxis
