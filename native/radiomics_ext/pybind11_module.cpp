#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "first_order.h"
#include "texture.h"

namespace py = pybind11;
using namespace medaxis::radiomics_ext;

PYBIND11_MODULE(medaxis_radiomics, module)
{
    module.doc() = "MedAxis first-order radiomics features.";
    module.def(
        "first_order_features",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values)
        {
            auto flat = values.reshape({values.size()});
            const auto* begin = static_cast<const double*>(flat.data());
            return first_order_features(std::vector<double>(begin, begin + flat.size()));
        },
        py::arg("values"));

    module.def(
        "texture_features",
        &texture_features,
        py::arg("values"),
        py::arg("levels") = 32,
        py::arg("families") = std::vector<std::string>{"glcm", "glrlm", "glszm", "gldm", "ngtdm"},
        "Texture radiomics: GLCM(24) + GLRLM(16) + GLSZM(16) + GLDM(14) + NGTDM(5).");
}
