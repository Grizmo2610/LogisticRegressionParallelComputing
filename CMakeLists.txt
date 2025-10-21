cmake_minimum_required(VERSION 3.10)
project(main CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})
