#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace medaxis
{
namespace cgal_ext
{

/// Flat triangle mesh representation used across the CGAL module boundary.
struct MeshData
{
    std::vector<double>    vertices; ///< flat xyz triplets
    std::vector<std::size_t> faces; ///< flat index triplets into vertices
    std::vector<std::size_t> tets;  ///< flat index quadruplets (delaunay only)
};

/// Poisson surface reconstruction from an oriented point cloud.
/// points / normals: flat xyz triplets, same length.
/// Note: marching_cubes_surface was removed — CGAL 6.0.1 ships no
/// <CGAL/marching_cubes.h>; the Python wrapper falls back to VTK.
MeshData poisson_reconstruct(const std::vector<double>& points,
                             const std::vector<double>& normals,
                             double smoothing_angle_deg = 30.0);

/// Advancing-front reconstruction from an unoriented point cloud.
MeshData advancing_front_reconstruct(const std::vector<double>& points);

/// 3D Delaunay tetrahedralization; tet indices returned in MeshData::tets.
MeshData delaunay_triangulate(const std::vector<double>& points);

} // namespace cgal_ext
} // namespace medaxis
