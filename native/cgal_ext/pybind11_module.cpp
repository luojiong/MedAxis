#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mesh_reconstruct.h"
#include "surface_processing.h"
#include "curvature.h"

namespace py = pybind11;

using namespace medaxis::cgal_ext;

namespace
{

std::vector<double> to_flat(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    auto flat = arr.reshape({arr.size()});
    const double* ptr = static_cast<const double*>(flat.data());
    return std::vector<double>(ptr, ptr + flat.size());
}

std::vector<float> to_flat_float(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto flat = arr.reshape({arr.size()});
    const float* ptr = static_cast<const float*>(flat.data());
    return std::vector<float>(ptr, ptr + flat.size());
}

py::array_t<double> vertices_array(const MeshData& m)
{
    return py::array_t<double>({static_cast<py::ssize_t>(m.vertices.size() / 3),
                                py::ssize_t(3)},
                               m.vertices.data());
}

py::array_t<std::size_t> faces_array(const MeshData& m)
{
    return py::array_t<std::size_t>({static_cast<py::ssize_t>(m.faces.size() / 3),
                                     py::ssize_t(3)},
                                    m.faces.data());
}

} // namespace

PYBIND11_MODULE(medaxis_cgal, m)
{
    m.doc() =
        "MedAxis CGAL module.\n"
        "网格重建 / 处理 / 曲率分析（CGAL 后端）。\n"
        "Mesh reconstruction, processing and curvature analysis backed by CGAL.";

    // --------------------------------------------------------------
    // MeshData
    // --------------------------------------------------------------
    py::class_<MeshData>(m, "MeshData",
        "Flat triangle mesh (vertices: Nx3 float64, faces: Mx3 uint64).\n"
        "扁平三角网格：顶点 Nx3，面片 Mx3。")
        .def(py::init<>())
        .def_readwrite("vertices", &MeshData::vertices)
        .def_readwrite("faces", &MeshData::faces)
        .def_readwrite("tets", &MeshData::tets)
        .def("vertices_array", [](const MeshData& m) { return vertices_array(m); },
             "Vertices as an (N, 3) numpy array. 顶点数组。")
        .def("faces_array", [](const MeshData& m) { return faces_array(m); },
             "Faces as an (M, 3) numpy array. 面片索引数组。");

    py::class_<PrincipalCurvatures>(m, "PrincipalCurvatures",
        "Per-vertex principal curvature results. 逐顶点主曲率结果。")
        .def(py::init<>())
        .def_readwrite("k1", &PrincipalCurvatures::k1)
        .def_readwrite("k2", &PrincipalCurvatures::k2)
        .def_readwrite("dir1", &PrincipalCurvatures::dir1)
        .def_readwrite("dir2", &PrincipalCurvatures::dir2);

    py::enum_<SmoothMode>(m, "SmoothMode")
        .value("LAPLACIAN", SmoothMode::Laplacian)
        .value("TAUBIN", SmoothMode::Taubin);

    py::enum_<SubdivideScheme>(m, "SubdivideScheme")
        .value("LOOP", SubdivideScheme::Loop)
        .value("CATMULL_CLARK", SubdivideScheme::CatmullClark);

    // --------------------------------------------------------------
    // Reconstruction
    // --------------------------------------------------------------
    // Note: marching_cubes_surface was removed (CGAL 6.0.1 has no
    // <CGAL/marching_cubes.h>); geometry.cgal_wrapper falls back to VTK.
    m.def("poisson_reconstruct",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
           py::array_t<double, py::array::c_style | py::array::forcecast> normals,
           double smoothing_angle)
        {
            return poisson_reconstruct(to_flat(points), to_flat(normals), smoothing_angle);
        },
        py::arg("points"), py::arg("normals"), py::arg("smoothing_angle_deg") = 30.0,
        "Poisson surface reconstruction from an oriented point cloud.\n"
        "基于带法向点云的泊松重建。");

    m.def("advancing_front_reconstruct",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> points)
        {
            return advancing_front_reconstruct(to_flat(points));
        },
        py::arg("points"),
        "Advancing-front reconstruction from a point cloud.\n"
        "基于点云的前沿推进重建。");

    m.def("delaunay_triangulate",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> points)
        {
            return delaunay_triangulate(to_flat(points));
        },
        py::arg("points"),
        "3D Delaunay tetrahedralization.\n"
        "三维 Delaunay 四面体化。");

    // --------------------------------------------------------------
    // Surface processing
    // --------------------------------------------------------------
    m.def("mesh_smooth", &mesh_smooth,
        py::arg("mesh"), py::arg("mode"), py::arg("iterations") = 10,
        "Smooth a mesh (Laplacian / Taubin).\n"
        "网格平滑（拉普拉斯 / Taubin）。");

    m.def("mesh_decimate", &mesh_decimate,
        py::arg("mesh"), py::arg("target_face_count"),
        "Decimate a mesh to a target face count.\n"
        "网格抽稀到目标面片数。");

    m.def("mesh_fill_holes", &mesh_fill_holes,
        py::arg("mesh"), py::arg("max_hole_edges") = 0,
        "Fill boundary holes. 修补网格孔洞。");

    m.def("mesh_boolean_union", &mesh_boolean_union,
        py::arg("a"), py::arg("b"), "Boolean union. 布尔并集。");
    m.def("mesh_boolean_intersect", &mesh_boolean_intersect,
        py::arg("a"), py::arg("b"), "Boolean intersection. 布尔交集。");
    m.def("mesh_boolean_diff", &mesh_boolean_diff,
        py::arg("a"), py::arg("b"), "Boolean difference (a - b). 布尔差集。");

    m.def("mesh_subdivide", &mesh_subdivide,
        py::arg("mesh"), py::arg("scheme"), py::arg("iterations") = 1,
        "Subdivide a mesh (Loop / Catmull-Clark).\n"
        "网格细分（Loop / Catmull-Clark）。");

    // --------------------------------------------------------------
    // Curvature
    // --------------------------------------------------------------
    m.def("compute_mean_curvature", &compute_mean_curvature,
        py::arg("mesh"), py::arg("neighborhood_radius"),
        "Per-vertex mean curvature. 逐顶点平均曲率。");

    m.def("compute_gaussian_curvature", &compute_gaussian_curvature,
        py::arg("mesh"), py::arg("neighborhood_radius"),
        "Per-vertex Gaussian curvature. 逐顶点高斯曲率。");

    m.def("compute_principal_curvatures", &compute_principal_curvatures,
        py::arg("mesh"), py::arg("neighborhood_radius"),
        "Per-vertex principal curvatures and directions.\n"
        "逐顶点主曲率与主方向。");


}
