#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace medaxis
{
namespace native_ops
{

/// Filtering: gaussian, median, bilateral, anisotropic_diffusion, nl_means,
/// gradient_magnitude, laplacian, sobel, canny, zero_crossing, mean,
/// discrete_gaussian, normalize, rescale, cast, sharpen.
py::array_t<float> filter_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind,
    const py::dict& params);

/// Morphology: binary/grayscale erode, dilate, open, close, tophat,
/// bottomhat, gradient; hole_filling, thinning, distance_map,
/// connected_components. Binary kinds return uint8, grayscale float32.
py::array morphology_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind,
    const py::dict& params);

/// Advanced segmentation (uint8 label output): connected_threshold,
/// confidence_connected, neighborhood_connected, watershed, fast_marching,
/// geodesic_active_contour, shape_detection, threshold_segmentation,
/// laplacian_segmentation, kmeans, fcm, island_remove.
py::array_t<std::uint8_t> segment_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind,
    const py::dict& params);

/// Arithmetic (second operand is a constant or a volume array):
/// add, subtract, multiply, divide, mask, invert.
py::array_t<float> arith_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    const py::object& b,
    const std::string& kind,
    const py::dict& params);

/// Enhancement: clahe, histogram_equalization, window_level, log, gamma.
py::array_t<float> enhance_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind,
    const py::dict& params);

/// Geometry: resample, crop, pad, flip, permute.
py::array_t<float> geometry_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    const std::string& kind,
    const py::dict& params);

/// Registration: rigid, affine, bspline, demons, diffeomorphic_demons,
/// fast_symmetric_demons. Returns the warped moving volume.
py::array_t<float> register_pair(
    py::array_t<float, py::array::c_style | py::array::forcecast> fixed,
    py::array_t<float, py::array::c_style | py::array::forcecast> moving,
    const std::string& kind,
    const py::dict& params);

} // namespace native_ops
} // namespace medaxis
