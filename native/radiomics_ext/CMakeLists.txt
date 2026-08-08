# Lightweight first-order radiomics extension. More expensive feature families
# can be added without changing the Python module boundary.
pybind11_add_module(medaxis_radiomics
    first_order.cpp
    pybind11_module.cpp
)

target_include_directories(medaxis_radiomics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS medaxis_radiomics
    LIBRARY DESTINATION medaxis/_native
    RUNTIME DESTINATION medaxis/_native
)
