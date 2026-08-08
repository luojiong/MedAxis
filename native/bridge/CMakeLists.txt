cmake_minimum_required(VERSION 3.21)
project(native_bridge LANGUAGES CXX)

find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel)
find_package(ITK REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(medaxis_bridge
    data_bridge.cpp
    pybind11_bridge.cpp
)

target_link_libraries(medaxis_bridge PRIVATE
    VTK::CommonCore
    VTK::CommonDataModel
    ITKCommon
)

target_include_directories(medaxis_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
