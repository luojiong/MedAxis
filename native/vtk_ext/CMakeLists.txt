# =============================================================================
# MedAxis VTK extension filters - vtkMedAxisFilters
# =============================================================================
add_library(vtkMedAxisFilters
    vtkMedAxisResliceFilter.cxx
    vtkMedAxisCPRFilter.cxx
)

target_include_directories(vtkMedAxisFilters PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vtkMedAxisFilters PUBLIC
    ${VTK_LIBRARIES}
)

vtk_module_autoinit(
    TARGETS vtkMedAxisFilters
    MODULES ${VTK_LIBRARIES}
)

install(TARGETS vtkMedAxisFilters
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
