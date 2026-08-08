# =============================================================================
# MedAxis CGAL module - mesh reconstruction, processing, curvature
# =============================================================================
find_package(CGAL REQUIRED)

# CGAL curvature measures require Eigen (CONFIG mode: see native/CMakeLists.txt)
find_package(Eigen3 REQUIRED CONFIG)

pybind11_add_module(medaxis_cgal
    mesh_reconstruct.cpp
    surface_processing.cpp
    curvature.cpp
    pybind11_module.cpp
)

target_include_directories(medaxis_cgal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(medaxis_cgal PRIVATE
    CGAL::CGAL
    Eigen3::Eigen
)

install(TARGETS medaxis_cgal
    LIBRARY DESTINATION medaxis/_native
    RUNTIME DESTINATION medaxis/_native
)
