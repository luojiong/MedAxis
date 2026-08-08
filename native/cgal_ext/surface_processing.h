#pragma once

#include "mesh_reconstruct.h"

namespace medaxis
{
namespace cgal_ext
{

/// Smoothing algorithms.
enum class SmoothMode
{
    Laplacian, ///< uniform Laplacian smoothing (geometry only)
    Taubin     ///< scale-compensated Taubin smoothing (better shape preservation)
};

/// Subdivision schemes.
enum class SubdivideScheme
{
    Loop,
    CatmullClark
};

/// Smooth a triangle mesh in place.
MeshData mesh_smooth(const MeshData& mesh, SmoothMode mode, unsigned int iterations);

/// Decimate a triangle mesh to (approximately) target_face_count faces.
MeshData mesh_decimate(const MeshData& mesh, std::size_t target_face_count);

/// Fill all boundary holes up to max_hole_edges boundary edges each
/// (0 = no limit).
MeshData mesh_fill_holes(const MeshData& mesh, std::size_t max_hole_edges = 0);

/// Boolean operations on closed triangle meshes.
MeshData mesh_boolean_union(const MeshData& a, const MeshData& b);
MeshData mesh_boolean_intersect(const MeshData& a, const MeshData& b);
MeshData mesh_boolean_diff(const MeshData& a, const MeshData& b);

/// Subdivision.
MeshData mesh_subdivide(const MeshData& mesh, SubdivideScheme scheme,
                        unsigned int iterations);

} // namespace cgal_ext
} // namespace medaxis
