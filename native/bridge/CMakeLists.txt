cmake_minimum_required(VERSION 3.21)
project(native_bridge LANGUAGES CXX)

find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel)
find_package(ITK REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(medaxis_bridge
    data_bridge.cpp
    pybind11_bridge.cpp
)

# Link against the *pip* VTK wheel's DLLs (via generated import libraries)
# so the packaged app carries a single VTK runtime.  The conda VTK libs are
# named vtkCommonCore-9.6.lib and would pull a second VTK into the process.
set(MEDAXIS_PIP_VTK_LIBS "${CMAKE_SOURCE_DIR}/build-deps/pip_vtk_libs")
if(EXISTS "${MEDAXIS_PIP_VTK_LIBS}/vtkCommonCore-9.6.2.lib")
    target_link_libraries(medaxis_bridge PRIVATE
        "${MEDAXIS_PIP_VTK_LIBS}/vtkCommonCore-9.6.2.lib"
        "${MEDAXIS_PIP_VTK_LIBS}/vtkCommonDataModel-9.6.2.lib"
        "${MEDAXIS_PIP_VTK_LIBS}/vtkCommonMath-9.6.2.lib"
        "${MEDAXIS_PIP_VTK_LIBS}/vtksys-9.6.2.lib"
        ITKCommon
    )
else()
    message(WARNING "pip vtk import libs not found — falling back to conda VTK")
    target_link_libraries(medaxis_bridge PRIVATE
        VTK::CommonCore
        VTK::CommonDataModel
        ITKCommon
    )
endif()

# VTK config exposes includes via targets only (no VTK_INCLUDE_DIRS variable).
get_target_property(MEDAXIS_VTK_INCLUDES VTK::CommonCore INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(medaxis_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MEDAXIS_VTK_INCLUDES}
)
