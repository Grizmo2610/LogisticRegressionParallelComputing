cmake_minimum_required(VERSION 3.10)
project(Logistic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
include_directories(include)

# Find pybind11 for Python module build
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES "src/*.cpp")

# Uncomment this line if you want to build and run as a standalone C++ executable
# add_executable(Logistic ${SOURCES})

# Uncomment this line if you want to build as a Python module (.so)
pybind11_add_module(Logistic ${SOURCES})