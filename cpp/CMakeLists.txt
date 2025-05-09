cmake_minimum_required(VERSION 3.11)
project(read_fem_tetra4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(read_gmsh_cpp MODULE read_fem_tetra4.cpp)

target_compile_options(read_gmsh_cpp PRIVATE -O3 -ffast-math -march=native)
