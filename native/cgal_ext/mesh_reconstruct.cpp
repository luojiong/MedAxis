#include "mesh_reconstruct.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/advancing_front_surface_reconstruction.h>
#include <CGAL/property_map.h>
#include <CGAL/Delaunay_triangulation_3.h>

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

/// Extract vertices / faces from a CGAL Surface_mesh into MeshData.
MeshData extract_mesh(const SurfaceMesh& mesh)
{
    MeshData out;
    out.vertices.reserve(3 * mesh.number_of_vertices());
    out.faces.reserve(3 * mesh.number_of_faces());

    std::map<SurfaceMesh::Vertex_index, std::size_t> vmap;

    for (const auto v : mesh.vertices())
    {
        vmap[v] = out.vertices.size() / 3;
        const auto& p = mesh.point(v);
        out.vertices.push_back(static_cast<double>(p.x()));
        out.vertices.push_back(static_cast<double>(p.y()));
        out.vertices.push_back(static_cast<double>(p.z()));
    }

    for (const auto f : mesh.faces())
    {
        for (const auto v : vertices_around_face(mesh.halfedge(f), mesh))
        {
            out.faces.push_back(vmap.at(v));
        }
    }

    return out;
}

} // namespace

MeshData poisson_reconstruct(const std::vector<double>& points,
                             const std::vector<double>& normals,
                             double smoothing_angle_deg)
{
    if (points.size() % 3 != 0 || points.size() != normals.size())
    {
        throw std::runtime_error("poisson_reconstruct: invalid point/normal arrays");
    }

    using Pwn = std::pair<Point, Vector>;
    std::vector<Pwn> pwn;
    pwn.reserve(points.size() / 3);
    for (std::size_t i = 0; i + 2 < points.size(); i += 3)
    {
        pwn.emplace_back(Point(points[i], points[i + 1], points[i + 2]),
                         Vector(normals[i], normals[i + 1], normals[i + 2]));
    }

    SurfaceMesh mesh;
    // CGAL 6.x: poisson_surface_reconstruction_delaunay takes a mandatory
    // spacing (average point spacing) followed by positional mesh-size bounds.
    // Estimate spacing from the bounding-box diagonal (avoids a KD-tree).
    Point bbox_min = pwn.front().first, bbox_max = pwn.front().first;
    for (const auto& e : pwn)
    {
        const Point& p = e.first;
        bbox_min = Point((std::min)(bbox_min.x(), p.x()),
                         (std::min)(bbox_min.y(), p.y()),
                         (std::min)(bbox_min.z(), p.z()));
        bbox_max = Point((std::max)(bbox_max.x(), p.x()),
                         (std::max)(bbox_max.y(), p.y()),
                         (std::max)(bbox_max.z(), p.z()));
    }
    const Vector diag = bbox_max - bbox_min;
    const double diag_len = std::sqrt(diag.squared_length());
    const double spacing =
        diag_len / std::cbrt(static_cast<double>(pwn.size())) * 0.5;
    CGAL::poisson_surface_reconstruction_delaunay(
        pwn.begin(), pwn.end(),
        CGAL::First_of_pair_property_map<Pwn>(),
        CGAL::Second_of_pair_property_map<Pwn>(),
        mesh, spacing, smoothing_angle_deg);

    return extract_mesh(mesh);
}

MeshData advancing_front_reconstruct(const std::vector<double>& points)
{
    if (points.size() % 3 != 0)
    {
        throw std::runtime_error("advancing_front_reconstruct: invalid point array");
    }

    std::vector<Point> pts;
    pts.reserve(points.size() / 3);
    for (std::size_t i = 0; i + 2 < points.size(); i += 3)
    {
        pts.emplace_back(points[i], points[i + 1], points[i + 2]);
    }

    std::vector<std::array<std::size_t, 3>> facets;
    CGAL::advancing_front_surface_reconstruction(
        pts.cbegin(), pts.cend(), std::back_inserter(facets));

    MeshData out;
    out.vertices.reserve(points.size());
    out.vertices = points;
    out.faces.reserve(3 * facets.size());
    for (const auto& f : facets)
    {
        out.faces.push_back(f[0]);
        out.faces.push_back(f[1]);
        out.faces.push_back(f[2]);
    }
    return out;
}

MeshData delaunay_triangulate(const std::vector<double>& points)
{
    if (points.size() % 3 != 0)
    {
        throw std::runtime_error("delaunay_triangulate: invalid point array");
    }

    using Delaunay = CGAL::Delaunay_triangulation_3<Kernel>;
    Delaunay dt;

    for (std::size_t i = 0; i + 2 < points.size(); i += 3)
    {
        dt.insert(Point(points[i], points[i + 1], points[i + 2]));
    }

    MeshData out;
    out.vertices = points;

    // Map vertex handles to indices.
    std::map<Delaunay::Vertex_handle, std::size_t> vmap;
    for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
    {
        vmap[vit] = vmap.size();
    }

    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit)
    {
        for (int k = 0; k < 4; ++k)
        {
            out.tets.push_back(vmap.at(cit->vertex(k)));
        }
    }

    return out;
}

} // namespace cgal_ext
} // namespace medaxis
