#pragma once

#include <array>
#include <vector>

namespace medaxis
{
namespace occ_ext
{

using Point3 = std::array<double, 3>;

/// Local Frenet frame at a curve parameter.
struct FrenetFrame
{
    Point3 point;
    Point3 tangent;
    Point3 normal;
    Point3 binormal;
};

/// Fit a B-spline through the control points and sample `num_samples`
/// points at equal arc-length intervals.
std::vector<Point3> sample_curve_equal_arc_length(const std::vector<Point3>& control_points,
                                                  int num_samples);

/// Total arc length of the fitted curve.
double curve_arc_length(const std::vector<Point3>& control_points);

/// Frenet frames (point/tangent/normal/binormal) at normalized parameters
/// in [0, 1] along the fitted curve.
std::vector<FrenetFrame> frenet_frames(const std::vector<Point3>& control_points,
                                       const std::vector<double>& parameters);

} // namespace occ_ext
} // namespace medaxis
