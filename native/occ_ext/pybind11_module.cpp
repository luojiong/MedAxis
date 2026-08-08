#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "curve_sampling.h"
#include "surface_utils.h"

namespace py = pybind11;
using namespace medaxis::occ_ext;

PYBIND11_MODULE(medaxis_occ, module)
{
    module.doc() =
        "MedAxis OpenCASCADE geometry operations: B-spline curve and surface sampling.";

    py::class_<FrenetFrame>(module, "FrenetFrame")
        .def(py::init<>())
        .def_readwrite("point", &FrenetFrame::point)
        .def_readwrite("tangent", &FrenetFrame::tangent)
        .def_readwrite("normal", &FrenetFrame::normal)
        .def_readwrite("binormal", &FrenetFrame::binormal);

    py::class_<SurfaceSample>(module, "SurfaceSample")
        .def(py::init<>())
        .def_readwrite("point", &SurfaceSample::point)
        .def_readwrite("normal", &SurfaceSample::normal);

    module.def("sample_curve_equal_arc_length", &sample_curve_equal_arc_length,
               py::arg("control_points"), py::arg("num_samples"));
    module.def("curve_arc_length", &curve_arc_length, py::arg("control_points"));
    module.def("frenet_frames", &frenet_frames,
               py::arg("control_points"), py::arg("parameters"));

    module.def("sample_surface_grid", &sample_surface_grid,
               py::arg("grid_points"), py::arg("num_u_cols"), py::arg("num_v_rows"),
               py::arg("samples_u"), py::arg("samples_v"));
    module.def("closest_point_on_surface", &closest_point_on_surface,
               py::arg("grid_points"), py::arg("num_u_cols"), py::arg("num_v_rows"),
               py::arg("query"));
    module.def("evaluate_surface", &evaluate_surface,
               py::arg("grid_points"), py::arg("num_u_cols"), py::arg("num_v_rows"),
               py::arg("u"), py::arg("v"));
}
