# =============================================================================
# MedAxis ITK extension filters (header-only template library)
#
# Filter subdirectories:
#   filtering/    - smoothing / denoising wrappers (Bilateral, ...)
#   morphology/   - binary & grayscale morphology
#   segmentation/ - thresholding, region growing, watershed, level sets,
#                   graph cut, brush, flood fill, auto fill
#   registration/ - rigid, affine, BSpline, demons
#
# These are template classes (.h + .hxx) and are header-only; consumers
# link medaxis_itk_ext and include the headers directly.
# =============================================================================

file(GLOB_RECURSE MEDAXIS_ITK_EXT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/filtering/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/filtering/*.hxx
    ${CMAKE_CURRENT_SOURCE_DIR}/morphology/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/morphology/*.hxx
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation/*.hxx
    ${CMAKE_CURRENT_SOURCE_DIR}/registration/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/registration/*.hxx
)

add_library(medaxis_itk_ext INTERFACE)

target_include_directories(medaxis_itk_ext INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(medaxis_itk_ext INTERFACE
    ${ITK_LIBRARIES}
)

target_compile_features(medaxis_itk_ext INTERFACE cxx_std_17)

# Make headers visible in IDEs
add_custom_target(medaxis_itk_ext_headers SOURCES ${MEDAXIS_ITK_EXT_HEADERS})

pybind11_add_module(medaxis_itk
    pybind11_module.cpp
)

target_include_directories(medaxis_itk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation
)

target_link_libraries(medaxis_itk PRIVATE
    medaxis_itk_ext
    ${ITK_LIBRARIES}
)

install(TARGETS medaxis_itk
    LIBRARY DESTINATION medaxis/_native
    RUNTIME DESTINATION medaxis/_native
)
