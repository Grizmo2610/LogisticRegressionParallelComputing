cmake_minimum_required(VERSION 3.18)
project(logreg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(pybind11 REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_library(logreg MODULE
        kernel.cu
        binding.cpp
)


set_target_properties(logreg  PROPERTIES
        PREFIX ""
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(logreg PRIVATE pybind11::module)
