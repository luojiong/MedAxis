cmake_minimum_required(VERSION 3.21)

project(MedAxis VERSION 0.1.0 LANGUAGES CXX)

option(MEDAXIS_BUILD_NATIVE "Build MedAxis native extensions" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MEDAXIS_BUILD_NATIVE)
    add_subdirectory(native)
endif()
