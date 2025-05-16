cmake_minimum_required(VERSION 3.11)
project(tetra4_fem_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)  

pybind11_add_module(tetra4_fem_core MODULE tetra4_fem_core.cpp)

target_compile_options(tetra4_fem_core PRIVATE -O3 -march=native -ffast-math -fopenmp)
target_link_libraries(tetra4_fem_core PRIVATE OpenMP::OpenMP_CXX) 
