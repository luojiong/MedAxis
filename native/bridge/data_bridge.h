#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <itkImage.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vtkImageData.h>
#include <vtkMatrix3x3.h>
#include <vtkSmartPointer.h>

namespace py = pybind11;

namespace medaxis {

struct VolumeInfo {
    std::vector<double> spacing;
    std::vector<double> origin;
    std::vector<double> direction;
    std::vector<int> dimensions;
    int scalar_type = 0;
    std::size_t num_components = 1;
    std::string modality;
};

class DataBridge {
public:
    using ITKImageType = itk::Image<float, 3>;

    static DataBridge& instance();

    int create_volume(py::array_t<float> data, const VolumeInfo& info);
    int create_volume_from_slices(
        const std::vector<py::array_t<float>>& slices,
        const VolumeInfo& info);

    vtkSmartPointer<vtkImageData> get_vtk_image(int handle);
    ITKImageType::Pointer get_itk_image(int handle);
    py::array get_numpy_array(int handle);

    void release_volume(int handle);
    VolumeInfo get_volume_info(int handle) const;

    std::vector<double> world_to_voxel(int handle, double wx, double wy, double wz);
    std::vector<double> voxel_to_world(int handle, int vx, int vy, int vz);

private:
    struct VolumeRecord {
        vtkSmartPointer<vtkImageData> vtk_image;
        ITKImageType::Pointer itk_image;
        py::array numpy_array;
        VolumeInfo info;
        bool dirty_itk = true;
        bool dirty_numpy = true;
    };

    DataBridge() = default;

    int next_handle();
    VolumeRecord& record(int handle);

    std::unordered_map<int, VolumeRecord> volumes_;
    int next_handle_ = 1;
    mutable std::mutex mutex_;
};

}  // namespace medaxis
