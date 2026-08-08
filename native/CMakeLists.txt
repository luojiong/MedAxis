cmake_minimum_required(VERSION 3.21)
project(medaxis_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──
find_package(VTK 9.4 QUIET COMPONENTS
    CommonCore CommonDataModel CommonMath
    ImagingCore ImagingGeneral ImagingHybrid
    FiltersCore FiltersGeneral FiltersHybrid
    RenderingCore RenderingVolumeOpenGL2
    InteractionStyle InteractionWidgets
)
find_package(ITK 5.4 QUIET COMPONENTS
    ITKCommon ITKImageFilterBase ITKImageIntensity ITKImageGradient
    ITKImageGrid ITKThresholding ITKStatistics
    ITKRegionGrowing ITKWatersheds ITKLevelSets
    ITKMathematicalMorphology ITKBinaryMathematicalMorphology
    ITKConnectedComponents ITKDistanceMap ITKRegistrationCommon
    ITKSmoothing ITKAnisotropicSmoothing
)
find_package(CGAL 5.6 QUIET)
# conda-forge eigen 5.x: 版本宏移到 Eigen/Version, CGAL 的 FindEigen3 模块解析失败,
# 强制 CONFIG 模式 (配置已复制到 lib/cmake/Eigen3)
find_package(Eigen3 QUIET CONFIG)
# conda-forge occt 用 ExactVersion 策略, 7.8.1 不匹配 7.8
find_package(OpenCASCADE QUIET)
find_package(pybind11 2.11 QUIET)

# ── MSVC specific ──
if(MSVC)
    add_compile_options(/bigobj /MP /utf-8)
endif()

# ── Subdirectories ──
if(VTK_FOUND AND ITK_FOUND AND pybind11_FOUND)
    add_subdirectory(bridge)
else()
    message(WARNING "Skipping medaxis_bridge: VTK, ITK, and pybind11 are required")
endif()

# VTK extensions
if(VTK_FOUND)
    add_subdirectory(vtk_ext)
else()
    message(WARNING "Skipping VTK extensions: VTK development package was not found")
endif()

# ITK extensions
if(ITK_FOUND AND pybind11_FOUND)
    add_subdirectory(itk_ext)
else()
    message(WARNING "Skipping ITK extensions: ITK and pybind11 are required")
endif()

# CGAL extensions (optional)
if(CGAL_FOUND AND Eigen3_FOUND AND pybind11_FOUND)
    add_subdirectory(cgal_ext)
elseif(CGAL_FOUND OR Eigen3_FOUND)
    message(WARNING "Skipping CGAL extensions: CGAL, Eigen3, and pybind11 are required")
endif()

# OCC extensions (optional)
if(OpenCASCADE_FOUND AND pybind11_FOUND)
    add_subdirectory(occ_ext)
endif()

# Radiomics
if(pybind11_FOUND)
    add_subdirectory(radiomics_ext)
else()
    message(WARNING "Skipping radiomics extension: pybind11 was not found")
endif()
