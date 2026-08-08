# =============================================================================
# MedAxis OpenCASCADE module - curve sampling & surface utilities
# =============================================================================
pybind11_add_module(medaxis_occ
    curve_sampling.cpp
    surface_utils.cpp
    pybind11_module.cpp
)

target_include_directories(medaxis_occ PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCASCADE_INCLUDE_DIR}
)

# Foundation classes used by the module.
target_link_libraries(medaxis_occ PRIVATE
    TKernel
    TKMath
    TKG2d
    TKG3d
    TKGeomBase
    TKGeomAlgo
    TKBRep
)

install(TARGETS medaxis_occ
    LIBRARY DESTINATION medaxis/_native
    RUNTIME DESTINATION medaxis/_native
)
