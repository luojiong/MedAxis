#include "curvature.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>

#include <stdexcept>

namespace medaxis
{
namespace cgal_ext
{

namespace
{

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using SurfaceMesh = CGAL::Surface_mesh<Point>;
namespace PMP = CGAL::Polygon_mesh_processing;

SurfaceMesh build_mesh(const MeshData& data)
{
    SurfaceMesh mesh;
    const std::size_t nv = data.vertices.size() / 3;

    for (std::size_t i = 0; i + 2 < data.vertices.size(); i += 3)
    {
        mesh.add_vertex(Point(data.vertices[i], data.vertices[i + 1], data.vertices[i + 2]));
    }

    for (std::size_t f = 0; f + 2 < data.faces.size(); f += 3)
    {
        const std::size_t i0 = data.faces[f];
        const std::size_t i1 = data.faces[f + 1];
        const std::size_t i2 = data.faces[f + 2];
        if (i0 >= nv || i1 >= nv || i2 >= nv)
        {
            throw std::runtime_error("build_mesh: face index out of range");
        }
        mesh.add_face(SurfaceMesh::Vertex_index(static_cast<std::uint32_t>(i0)),
                      SurfaceMesh::Vertex_index(static_cast<std::uint32_t>(i1)),
                      SurfaceMesh::Vertex_index(static_cast<std::uint32_t>(i2)));
    }

    return mesh;
}

/// Per-vertex scalar values in `sm.vertices()` iteration order.
std::vector<double> collect_scalars(
    const SurfaceMesh& sm,
    const SurfaceMesh::Property_map<SurfaceMesh::Vertex_index, double>& values)
{
    std::vector<double> out(sm.number_of_vertices(), 0.0);
    for (const auto v : sm.vertices())
    {
        out[v.idx()] = values[v];
    }
    return out;
}

} // namespace

std::vector<double> compute_mean_curvature(const MeshData& mesh,
                                           double neighborhood_radius)
{
    SurfaceMesh sm = build_mesh(mesh);
    auto hmap = sm.add_property_map<SurfaceMesh::Vertex_index, double>("v:mean_curvature", 0.0).first;
    auto gmap = sm.add_property_map<SurfaceMesh::Vertex_index, double>("v:gaussian_curvature", 0.0).first;

    PMP::interpolated_corrected_curvatures(
        sm,
        CGAL::parameters::vertex_mean_curvature_map(hmap)
            .vertex_Gaussian_curvature_map(gmap)
            .neighbor_radius(neighborhood_radius));

    return collect_scalars(sm, hmap);
}

std::vector<double> compute_gaussian_curvature(const MeshData& mesh,
                                               double neighborhood_radius)
{
    SurfaceMesh sm = build_mesh(mesh);
    auto hmap = sm.add_property_map<SurfaceMesh::Vertex_index, double>("v:mean_curvature", 0.0).first;
    auto gmap = sm.add_property_map<SurfaceMesh::Vertex_index, double>("v:gaussian_curvature", 0.0).first;

    PMP::interpolated_corrected_curvatures(
        sm,
        CGAL::parameters::vertex_mean_curvature_map(hmap)
            .vertex_Gaussian_curvature_map(gmap)
            .neighbor_radius(neighborhood_radius));

    return collect_scalars(sm, gmap);
}

PrincipalCurvatures compute_principal_curvatures(const MeshData& mesh,
                                                 double neighborhood_radius)
{
    SurfaceMesh sm = build_mesh(mesh);

    auto pmap = sm.add_property_map<
        SurfaceMesh::Vertex_index,
        PMP::Principal_curvatures_and_directions<Kernel>>(
            "v:principal_curvatures", PMP::Principal_curvatures_and_directions<Kernel>())
        .first;

    PMP::interpolated_corrected_curvatures(
        sm,
        CGAL::parameters::vertex_principal_curvatures_and_directions_map(pmap)
            .neighbor_radius(neighborhood_radius));

    PrincipalCurvatures out;
    out.k1.reserve(sm.number_of_vertices());
    out.k2.reserve(sm.number_of_vertices());
    out.dir1.reserve(3 * sm.number_of_vertices());
    out.dir2.reserve(3 * sm.number_of_vertices());

    for (const auto v : sm.vertices())
    {
        const auto& pc = pmap[v];
        out.k1.push_back(static_cast<double>(pc.max_curvature));
        out.k2.push_back(static_cast<double>(pc.min_curvature));
        out.dir1.push_back(static_cast<double>(pc.max_direction.x()));
        out.dir1.push_back(static_cast<double>(pc.max_direction.y()));
        out.dir1.push_back(static_cast<double>(pc.max_direction.z()));
        out.dir2.push_back(static_cast<double>(pc.min_direction.x()));
        out.dir2.push_back(static_cast<double>(pc.min_direction.y()));
        out.dir2.push_back(static_cast<double>(pc.min_direction.z()));
    }

    return out;
}

} // namespace cgal_ext
} // namespace medaxis
