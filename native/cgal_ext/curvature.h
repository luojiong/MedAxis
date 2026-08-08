#pragma once

#include "mesh_reconstruct.h"

namespace medaxis
{
namespace cgal_ext
{

/// Per-vertex mean curvature (curvature-measures package).
std::vector<double> compute_mean_curvature(const MeshData& mesh,
                                            double neighborhood_radius);

/// Per-vertex Gaussian curvature.
std::vector<double> compute_gaussian_curvature(const MeshData& mesh,
                                               double neighborhood_radius);

/// Per-vertex principal curvatures and directions.
struct PrincipalCurvatures
{
    std::vector<double> k1;      ///< max curvature
    std::vector<double> k2;      ///< min curvature
    std::vector<double> dir1;    ///< flat xyz of k1 direction
    std::vector<double> dir2;    ///< flat xyz of k2 direction
};

PrincipalCurvatures compute_principal_curvatures(const MeshData& mesh,
                                                 double neighborhood_radius);

} // namespace cgal_ext
} // namespace medaxis
