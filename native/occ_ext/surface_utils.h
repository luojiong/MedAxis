#pragma once

#include <array>
#include <vector>

namespace medaxis
{
namespace occ_ext
{

using Point3 = std::array<double, 3>;

/// A point + normal sampled on a surface.
struct SurfaceSample
{
    Point3 point;
    Point3 normal;
};

/// Fit a B-spline surface through a regular grid of points
/// (grid is row-major, `num_u_cols` columns x `num_v_rows` rows) and
/// sample it on a dense `samples_u x samples_v` parameter grid.
std::vector<SurfaceSample> sample_surface_grid(const std::vector<Point3>& grid_points,
                                               int num_u_cols,
                                               int num_v_rows,
                                               int samples_u,
                                               int samples_v);

/// Closest point projection of `query` onto the fitted surface.
Point3 closest_point_on_surface(const std::vector<Point3>& grid_points,
                                int num_u_cols,
                                int num_v_rows,
                                const Point3& query);

/// Point-in-surface evaluation at normalized (u, v) in [0, 1]^2.
Point3 evaluate_surface(const std::vector<Point3>& grid_points,
                        int num_u_cols,
                        int num_v_rows,
                        double u,
                        double v);

} // namespace occ_ext
} // namespace medaxis
