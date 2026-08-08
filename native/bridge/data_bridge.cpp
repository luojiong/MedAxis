#include "data_bridge.h"

#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkMatrix4x4.h>
#include <itkImportImageFilter.h>

#include <cstring>
#include <stdexcept>

namespace medaxis {

namespace {

void validate_volume_info(const VolumeInfo& info) {
    if (info.dimensions.size() != 3 || info.spacing.size() != 3 ||
        info.origin.size() != 3 || info.direction.size() != 9) {
        throw std::invalid_argument(
            "DataBridge: dimensions, spacing, origin, and direction must be 3, 3, 3, and 9 elements");
    }
    if (info.num_components == 0) {
        throw std::invalid_argument("DataBridge: num_components must be positive");
    }
    for (int dimension : info.dimensions) {
        if (dimension <= 0) {
            throw std::invalid_argument("DataBridge: dimensions must be positive");
        }
    }
    for (double spacing : info.spacing) {
        if (spacing <= 0.0) {
            throw std::invalid_argument("DataBridge: spacing must be positive");
        }
    }
}

// Compute total number of voxels from VolumeInfo.
inline size_t voxel_count(const VolumeInfo& info) {
    size_t n = 1;
    for (int d : info.dimensions) n *= static_cast<size_t>(d);
    return n;
}

// Build the VolumeInfo fields onto a vtkImageData.
void apply_info_to_vtk(vtkImageData* img, const VolumeInfo& info) {
    img->SetDimensions(info.dimensions[0], info.dimensions[1], info.dimensions[2]);
    img->SetSpacing(info.spacing[0], info.spacing[1], info.spacing[2]);
    img->SetOrigin(info.origin[0], info.origin[1], info.origin[2]);

    // VTK 9.6: direction matrix is a vtkMatrix3x3 (vtkMatrix4x4 overload removed).
    vtkNew<vtkMatrix3x3> dir;
    dir->Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            dir->SetElement(r, c, info.direction[r * 3 + c]);
    img->SetDirectionMatrix(dir);
}

} // namespace

DataBridge& DataBridge::instance() {
    // Intentionally leaked: destroying the singleton during interpreter
    // shutdown races with VTK Python module teardown and crashes (classic
    // cross-language exit-order problem). The OS reclaims the memory.
    static DataBridge* s_instance = new DataBridge();
    return *s_instance;
}

int DataBridge::next_handle() {
    return next_handle_++;
}

DataBridge::VolumeRecord& DataBridge::record(int handle) {
    auto it = volumes_.find(handle);
    if (it == volumes_.end()) {
        throw std::runtime_error("DataBridge: unknown volume handle " + std::to_string(handle));
    }
    return it->second;
}

int DataBridge::create_volume(py::array_t<float> data, const VolumeInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    validate_volume_info(info);

    const size_t n = voxel_count(info) * info.num_components;

    // Ensure contiguous C-order float buffer.
    auto buf = data.request();
    if (buf.ndim < 1) throw std::runtime_error("DataBridge: expected at least 1-D array");
    if (static_cast<size_t>(buf.size) < n) {
        throw std::runtime_error("DataBridge: array too small for declared volume dimensions");
    }

    VolumeRecord rec;
    rec.info = info;

    // ------------------------------------------------------------------
    // Allocate the single source-of-truth buffer inside VTK.
    // ------------------------------------------------------------------
    vtkNew<vtkFloatArray> scalars;
    scalars->SetNumberOfComponents(static_cast<int>(info.num_components));
    scalars->SetNumberOfTuples(static_cast<vtkIdType>(voxel_count(info)));
    std::memcpy(scalars->GetVoidPointer(0), buf.ptr, n * sizeof(float));

    vtkNew<vtkImageData> img;
    apply_info_to_vtk(img, info);
    img->GetPointData()->SetScalars(scalars);

    rec.vtk_image = img;

    const int handle = next_handle();
    volumes_.emplace(handle, std::move(rec));
    return handle;
}

int DataBridge::create_volume_from_slices(const std::vector<py::array_t<float>>& slices,
                                          const VolumeInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    validate_volume_info(info);

    if (slices.empty()) throw std::runtime_error("DataBridge: no slices provided");
    if (slices.size() != static_cast<size_t>(info.dimensions[2])) {
        throw std::runtime_error("DataBridge: slice count does not match the z dimension");
    }

    const size_t per_slice = static_cast<size_t>(info.dimensions[0]) *
                             static_cast<size_t>(info.dimensions[1]) *
                             info.num_components;

    VolumeRecord rec;
    rec.info = info;

    vtkNew<vtkFloatArray> scalars;
    scalars->SetNumberOfComponents(static_cast<int>(info.num_components));
    scalars->SetNumberOfTuples(static_cast<vtkIdType>(voxel_count(info)));
    float* dst = static_cast<float*>(scalars->GetVoidPointer(0));

    size_t slice_index = 0;
    for (const auto& slice : slices) {
        auto buf = slice.request();
        if (static_cast<size_t>(buf.size) < per_slice) {
            throw std::runtime_error("DataBridge: slice " + std::to_string(slice_index) +
                                     " too small");
        }
        std::memcpy(dst + slice_index * per_slice, buf.ptr, per_slice * sizeof(float));
        ++slice_index;
    }

    vtkNew<vtkImageData> img;
    apply_info_to_vtk(img, info);
    img->GetPointData()->SetScalars(scalars);

    rec.vtk_image = img;

    const int handle = next_handle();
    volumes_.emplace(handle, std::move(rec));
    return handle;
}

vtkSmartPointer<vtkImageData> DataBridge::get_vtk_image(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    return record(handle).vtk_image;
}

DataBridge::ITKImageType::Pointer DataBridge::get_itk_image(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    VolumeRecord& rec = record(handle);

    if (!rec.itk_image || rec.dirty_itk) {
        const auto& info = rec.info;
        const size_t n = voxel_count(info);
        float* buffer = static_cast<float*>(
            rec.vtk_image->GetPointData()->GetScalars()->GetVoidPointer(0));

        using ImportFilterType = itk::ImportImageFilter<float, 3>;
        auto importer = ImportFilterType::New();

        ImportFilterType::SizeType size;
        size[0] = static_cast<unsigned long>(info.dimensions[0]);
        size[1] = static_cast<unsigned long>(info.dimensions[1]);
        size[2] = static_cast<unsigned long>(info.dimensions[2]);

        ImportFilterType::IndexType start;
        start.Fill(0);

        ImportFilterType::RegionType region;
        region.SetIndex(start);
        region.SetSize(size);
        importer->SetRegion(region);

        ITKImageType::SpacingType spacing;
        spacing[0] = info.spacing[0];
        spacing[1] = info.spacing[1];
        spacing[2] = info.spacing[2];
        importer->SetSpacing(spacing);

        ITKImageType::PointType origin;
        origin[0] = info.origin[0];
        origin[1] = info.origin[1];
        origin[2] = info.origin[2];
        importer->SetOrigin(origin);

        ITKImageType::DirectionType direction;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                direction(r, c) = info.direction[r * 3 + c];
        importer->SetDirection(direction);

        // ITK does NOT own the buffer; VTK image keeps it alive.
        const bool let_itk_manage_memory = false;
        importer->SetImportPointer(buffer, static_cast<long>(n), let_itk_manage_memory);
        importer->Update();

        rec.itk_image = importer->GetOutput();
        // Disconnect from pipeline so the image outlives the filter.
        rec.itk_image->DisconnectPipeline();
        rec.dirty_itk = false;
    }
    return rec.itk_image;
}

py::array DataBridge::get_numpy_array(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    VolumeRecord& rec = record(handle);

    if (!rec.numpy_array || rec.dirty_numpy) {
        const auto& info = rec.info;
        float* buffer = static_cast<float*>(
            rec.vtk_image->GetPointData()->GetScalars()->GetVoidPointer(0));

        std::vector<py::ssize_t> shape;
        if (info.num_components > 1) {
            shape = {static_cast<py::ssize_t>(info.dimensions[2]),
                     static_cast<py::ssize_t>(info.dimensions[1]),
                     static_cast<py::ssize_t>(info.dimensions[0]),
                     static_cast<py::ssize_t>(info.num_components)};
        } else {
            shape = {static_cast<py::ssize_t>(info.dimensions[2]),
                     static_cast<py::ssize_t>(info.dimensions[1]),
                     static_cast<py::ssize_t>(info.dimensions[0])};
        }

        // The NumPy view may outlive the bridge record. Hold one VTK reference
        // in the capsule so releasing the handle cannot leave a dangling view.
        vtkImageData* keep_alive = rec.vtk_image.GetPointer();
        keep_alive->Register(nullptr);
        py::capsule owner(keep_alive, [](void* pointer) {
            static_cast<vtkImageData*>(pointer)->Delete();
        });

        rec.numpy_array = py::array_t<float>(shape, buffer, owner);
        rec.dirty_numpy = false;
    }
    return rec.numpy_array;
}

void DataBridge::release_volume(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    volumes_.erase(handle);
}

VolumeInfo DataBridge::get_volume_info(int handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = volumes_.find(handle);
    if (it == volumes_.end()) {
        throw std::runtime_error("DataBridge: unknown volume handle " + std::to_string(handle));
    }
    return it->second.info;
}

std::vector<double> DataBridge::world_to_voxel(int handle, double wx, double wy, double wz) {
    std::lock_guard<std::mutex> lock(mutex_);
    const VolumeInfo& info = record(handle).info;

    // world = origin + D * diag(spacing) * voxel
    // voxel = diag(1/spacing) * D^T * (world - origin)
    const double dx = wx - info.origin[0];
    const double dy = wy - info.origin[1];
    const double dz = wz - info.origin[2];

    const double* d = info.direction.data(); // row-major 3x3

    std::vector<double> v(3);
    v[0] = (d[0] * dx + d[3] * dy + d[6] * dz) / info.spacing[0];
    v[1] = (d[1] * dx + d[4] * dy + d[7] * dz) / info.spacing[1];
    v[2] = (d[2] * dx + d[5] * dy + d[8] * dz) / info.spacing[2];
    return v;
}

std::vector<double> DataBridge::voxel_to_world(int handle, int vx, int vy, int vz) {
    std::lock_guard<std::mutex> lock(mutex_);
    const VolumeInfo& info = record(handle).info;

    const double sx = vx * info.spacing[0];
    const double sy = vy * info.spacing[1];
    const double sz = vz * info.spacing[2];
    const double* d = info.direction.data(); // row-major 3x3

    std::vector<double> w(3);
    w[0] = info.origin[0] + d[0] * sx + d[1] * sy + d[2] * sz;
    w[1] = info.origin[1] + d[3] * sx + d[4] * sy + d[5] * sz;
    w[2] = info.origin[2] + d[6] * sx + d[7] * sy + d[8] * sz;
    return w;
}

} // namespace medaxis
