#include "surface_utils.h"

#include <Geom_BSplineSurface.hxx>
#include <GeomAPI_PointsToBSplineSurface.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomLProp_SLProps.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>

#include <stdexcept>

namespace medaxis
{
namespace occ_ext
{

namespace
{

/// Fit a B-spline surface through a row-major point grid.
Handle(Geom_BSplineSurface) fit_surface(const std::vector<Point3>& grid_points,
                                        int num_u_cols,
                                        int num_v_rows)
{
    if (num_u_cols < 2 || num_v_rows < 2)
    {
        throw std::runtime_error("fit_surface: grid must be at least 2x2");
    }
    if (static_cast<int>(grid_points.size()) != num_u_cols * num_v_rows)
    {
        throw std::runtime_error("fit_surface: grid_points size mismatch");
    }

    TColgp_Array2OfPnt grid(1, num_v_rows, 1, num_u_cols);
    for (int v = 0; v < num_v_rows; ++v)
    {
        for (int u = 0; u < num_u_cols; ++u)
        {
            const auto& p = grid_points[static_cast<std::size_t>(v * num_u_cols + u)];
            grid.SetValue(v + 1, u + 1, gp_Pnt(p[0], p[1], p[2]));
        }
    }

    GeomAPI_PointsToBSplineSurface approximator(grid, 3, 3);
    if (!approximator.IsDone())
    {
        throw std::runtime_error("fit_surface: B-spline surface fitting failed");
    }
    return approximator.Surface();
}

} // namespace

std::vector<SurfaceSample> sample_surface_grid(const std::vector<Point3>& grid_points,
                                               int num_u_cols,
                                               int num_v_rows,
                                               int samples_u,
                                               int samples_v)
{
    if (samples_u < 2 || samples_v < 2)
    {
        throw std::runtime_error("sample_surface_grid: need >= 2 samples per axis");
    }

    Handle(Geom_BSplineSurface) surface = fit_surface(grid_points, num_u_cols, num_v_rows);

    const Standard_Real u0 = surface->UKnot(1);
    const Standard_Real u1 = surface->UKnot(surface->NbUKnots());
    const Standard_Real v0 = surface->VKnot(1);
    const Standard_Real v1 = surface->VKnot(surface->NbVKnots());

    std::vector<SurfaceSample> samples;
    samples.reserve(static_cast<std::size_t>(samples_u) * samples_v);

    for (int j = 0; j < samples_v; ++j)
    {
        const Standard_Real v = v0 + (v1 - v0) * j / (samples_v - 1);
        for (int i = 0; i < samples_u; ++i)
        {
            const Standard_Real u = u0 + (u1 - u0) * i / (samples_u - 1);

            SurfaceSample s;
            gp_Pnt p;
            surface->D0(u, v, p);
            s.point = {p.X(), p.Y(), p.Z()};

            GeomLProp_SLProps props(surface, u, v, 1, 1.0e-8);
            if (props.IsNormalDefined())
            {
                const gp_Dir n = props.Normal();
                s.normal = {n.X(), n.Y(), n.Z()};
            }
            else
            {
                s.normal = {0.0, 0.0, 1.0};
            }

            samples.push_back(s);
        }
    }

    return samples;
}

Point3 closest_point_on_surface(const std::vector<Point3>& grid_points,
                                int num_u_cols,
                                int num_v_rows,
                                const Point3& query)
{
    Handle(Geom_BSplineSurface) surface = fit_surface(grid_points, num_u_cols, num_v_rows);

    GeomAPI_ProjectPointOnSurf projector(gp_Pnt(query[0], query[1], query[2]), surface);
    if (projector.NbPoints() < 1)
    {
        throw std::runtime_error("closest_point_on_surface: projection failed");
    }

    const gp_Pnt nearest = projector.NearestPoint();
    return {nearest.X(), nearest.Y(), nearest.Z()};
}

Point3 evaluate_surface(const std::vector<Point3>& grid_points,
                        int num_u_cols,
                        int num_v_rows,
                        double u,
                        double v)
{
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
    {
        throw std::runtime_error("evaluate_surface: (u, v) out of [0, 1]^2");
    }

    Handle(Geom_BSplineSurface) surface = fit_surface(grid_points, num_u_cols, num_v_rows);

    const Standard_Real u0 = surface->UKnot(1);
    const Standard_Real u1 = surface->UKnot(surface->NbUKnots());
    const Standard_Real v0 = surface->VKnot(1);
    const Standard_Real v1 = surface->VKnot(surface->NbVKnots());

    gp_Pnt p;
    surface->D0(u0 + u * (u1 - u0), v0 + v * (v1 - v0), p);
    return {p.X(), p.Y(), p.Z()};
}

} // namespace occ_ext
} // namespace medaxis
