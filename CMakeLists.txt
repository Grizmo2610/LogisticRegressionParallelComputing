cmake_minimum_required(VERSION 3.10)
project(Logistic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
include_directories(include)

# Find pybind11 for Python module build
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP library for parallel computing
find_package(OpenMP REQUIRED)  # Added

file(GLOB SOURCES "src/*.cpp")

# Uncomment this line if you want to build and run as a standalone C++ executable
# add_executable(Logistic ${SOURCES})

# Uncomment this line if you want to build as a Python module (.so or .pyd)
pybind11_add_module(Logistic ${SOURCES})

# Specify include directories for the target
target_include_directories(Logistic PRIVATE include)

# Link OpenMP to the target
target_link_libraries(Logistic PRIVATE OpenMP::OpenMP_CXX)  # Added

# Set properties for the target
set_target_properties(Logistic PROPERTIES
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
)
