#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "data_bridge.h"

namespace py = pybind11;
using namespace medaxis;

PYBIND11_MODULE(medaxis_bridge, m) {
    m.doc() =
        "MedAxis native data bridge.\n"
        "零拷贝桥接层：在 Python(numpy) / VTK / ITK 之间共享同一块体数据内存。\n"
        "Zero-copy bridge sharing one voxel buffer between numpy, VTK and ITK.";

    // ------------------------------------------------------------------
    // VolumeInfo
    // ------------------------------------------------------------------
    py::class_<VolumeInfo>(m, "VolumeInfo",
        "Volume metadata (spacing/origin/direction/dimensions).\n"
        "体数据元信息：间距、原点、方向矩阵、尺寸。")
        .def(py::init<>())
        .def_readwrite("spacing", &VolumeInfo::spacing,
            "Voxel spacing {sx, sy, sz}. 体素间距。")
        .def_readwrite("origin", &VolumeInfo::origin,
            "World origin {ox, oy, oz}. 世界坐标原点。")
        .def_readwrite("direction", &VolumeInfo::direction,
            "Direction cosine matrix, 9 floats row-major. 方向余弦矩阵（行优先，9 个元素）。")
        .def_readwrite("dimensions", &VolumeInfo::dimensions,
            "Voxel dimensions {nx, ny, nz}. 体素维度。")
        .def_readwrite("scalar_type", &VolumeInfo::scalar_type,
            "VTK scalar type enum (e.g. 10 = VTK_FLOAT). VTK 标量类型枚举。")
        .def_readwrite("num_components", &VolumeInfo::num_components,
            "Number of scalar components per voxel. 每个体素的分量数。")
        .def("__repr__", [](const VolumeInfo& v) {
            return "VolumeInfo(dims=[" + std::to_string(v.dimensions[0]) + ", " +
                   std::to_string(v.dimensions[1]) + ", " +
                   std::to_string(v.dimensions[2]) + "])";
        });

    // ------------------------------------------------------------------
    // DataBridge
    // ------------------------------------------------------------------
    py::class_<DataBridge>(m, "DataBridge",
        "Singleton zero-copy bridge between numpy, VTK and ITK.\n"
        "单例零拷贝数据桥（numpy / VTK / ITK 共享内存）。")
        .def_static("instance", &DataBridge::instance,
            py::return_value_policy::reference,
            "Get the process-wide singleton. 获取进程级单例。")
        .def("create_volume", &DataBridge::create_volume,
            py::arg("data"), py::arg("info"),
            "Create a managed volume from a float numpy array.\n"
            "从 float numpy 数组创建受管体数据，返回整数句柄。\n"
            "Returns: integer handle usable with the other methods.")
        .def("create_volume_from_slices", &DataBridge::create_volume_from_slices,
            py::arg("slices"), py::arg("info"),
            "Create a volume from a list of 2D float slices (z-stack).\n"
            "从 2D 切片列表（z 方向堆叠）创建体数据，返回句柄。")
        .def("get_vtk_image",
            [](DataBridge& self, int handle) {
                // Return as opaque capsule; Python layer wraps via vtk python.
                vtkImageData* img = self.get_vtk_image(handle).GetPointer();
                return py::capsule(static_cast<void*>(img), "vtkImageData");
            },
            py::arg("handle"),
            "Get the VTK vtkImageData for a handle (opaque capsule).\n"
            "获取句柄对应的 vtkImageData（以 capsule 返回，供 vtk-python 包装）。")
        .def("get_itk_image",
            [](DataBridge& self, int handle) {
                auto img = self.get_itk_image(handle);
                return py::capsule(static_cast<void*>(img.GetPointer()), "itkImage");
            },
            py::arg("handle"),
            "Get the ITK image for a handle (opaque capsule).\n"
            "获取句柄对应的 ITK 图像（以 capsule 返回）。")
        .def("get_numpy_array", &DataBridge::get_numpy_array,
            py::arg("handle"),
            "Get a zero-copy numpy view of the volume buffer.\n"
            "获取体数据缓冲区的零拷贝 numpy 视图。")
        .def("release_volume", &DataBridge::release_volume,
            py::arg("handle"),
            "Release all resources for a handle. 释放句柄对应的全部资源。")
        .def("get_volume_info", &DataBridge::get_volume_info,
            py::arg("handle"),
            "Query metadata for a handle. 查询句柄对应的元信息。")
        .def("world_to_voxel", &DataBridge::world_to_voxel,
            py::arg("handle"), py::arg("wx"), py::arg("wy"), py::arg("wz"),
            "Convert world coordinates to (fractional) voxel indices.\n"
            "世界坐标 → 体素坐标（可为小数）。")
        .def("voxel_to_world", &DataBridge::voxel_to_world,
            py::arg("handle"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
            "Convert voxel indices to world coordinates.\n"
            "体素坐标 → 世界坐标。");
}
