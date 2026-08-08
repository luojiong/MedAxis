#include "surface_processing.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Subdivision_method_3.h>
// CGAL 6.x: decimation lives in Surface_mesh_simplification (PMP/decimate.h was removed)
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>

#include <cstdio>
#include <stdexcept>

namespace medaxis
{
namespace cgal_ext
{

namespace
{

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
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

MeshData boolean_op(const MeshData& a, const MeshData& b, int op)
{
    SurfaceMesh ma = build_mesh(a);
    SurfaceMesh mb = build_mesh(b);
    SurfaceMesh out;

    switch (op)
    {
        case 0:
            PMP::corefine_and_compute_union(ma, mb, out);
            break;
        case 1:
            PMP::corefine_and_compute_intersection(ma, mb, out);
            break;
        case 2:
            PMP::corefine_and_compute_difference(ma, mb, out);
            break;
        default:
            throw std::runtime_error("boolean_op: unknown operation");
    }

    return extract_mesh(out);
}

} // namespace

MeshData mesh_smooth(const MeshData& mesh, SmoothMode mode, unsigned int iterations)
{
    SurfaceMesh sm = build_mesh(mesh);

    if (mode == SmoothMode::Laplacian)
    {
        PMP::angle_and_area_smoothing(
            sm, CGAL::parameters::number_of_iterations(iterations));
    }
    else // Taubin (CGAL 6.x: smooth_shape takes a mandatory time step)
    {
        PMP::smooth_shape(sm, 1.0,
                          CGAL::parameters::number_of_iterations(iterations));
    }

    return extract_mesh(sm);
}

MeshData mesh_decimate(const MeshData& mesh, std::size_t target_face_count)
{
    SurfaceMesh sm = build_mesh(mesh);

    const std::size_t initial_faces = sm.number_of_faces();
    if (target_face_count == 0 || target_face_count >= initial_faces)
    {
        return extract_mesh(sm);
    }

    namespace SMS = CGAL::Surface_mesh_simplification;
    SMS::Count_ratio_stop_predicate<SurfaceMesh> stop(
        static_cast<double>(target_face_count) / static_cast<double>(initial_faces));
    SMS::edge_collapse(sm, stop,
        CGAL::parameters::get_cost(SMS::Edge_length_cost<SurfaceMesh>())
            .get_placement(SMS::Midpoint_placement<SurfaceMesh>()));

    sm.collect_garbage();
    return extract_mesh(sm);
}

MeshData mesh_fill_holes(const MeshData& mesh, std::size_t max_hole_edges)
{
    SurfaceMesh sm = build_mesh(mesh);

    // Collect one halfedge per boundary hole.
    std::vector<SurfaceMesh::Halfedge_index> holes;
    for (const auto h : sm.halfedges())
    {
        if (sm.is_border(h))
        {
            holes.push_back(h);
        }
    }

    for (const auto h : holes)
    {
        if (!sm.is_border(h))
        {
            continue; // already filled by a previous pass
        }
        if (max_hole_edges > 0)
        {
            std::size_t count = 0;
            for (const auto hh : halfedges_around_face(h, sm))
            {
                ++count;
                if (count > max_hole_edges)
                {
                    break;
                }
            }
            if (count > max_hole_edges)
            {
                continue;
            }
        }
        PMP::triangulate_hole(sm, h);
    }

    return extract_mesh(sm);
}

MeshData mesh_boolean_union(const MeshData& a, const MeshData& b)
{
    return boolean_op(a, b, 0);
}

MeshData mesh_boolean_intersect(const MeshData& a, const MeshData& b)
{
    return boolean_op(a, b, 1);
}

MeshData mesh_boolean_diff(const MeshData& a, const MeshData& b)
{
    return boolean_op(a, b, 2);
}

MeshData mesh_subdivide(const MeshData& mesh, SubdivideScheme scheme,
                        unsigned int iterations)
{
    SurfaceMesh sm = build_mesh(mesh);

    switch (scheme)
    {
        case SubdivideScheme::Loop:
            CGAL::Subdivision_method_3::Loop_subdivision(
                sm, CGAL::parameters::number_of_iterations(iterations));
            break;
        case SubdivideScheme::CatmullClark:
            CGAL::Subdivision_method_3::CatmullClark_subdivision(
                sm, CGAL::parameters::number_of_iterations(iterations));
            break;
    }

    return extract_mesh(sm);
}

} // namespace cgal_ext
} // namespace medaxis
