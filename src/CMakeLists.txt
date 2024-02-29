cmake_minimum_required(VERSION 3.4)
project(DistanceMetrics)

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(distance_metrics distance_metrics.cpp)
target_link_libraries(distance_metrics PRIVATE pybind11::module Eigen3::Eigen)

set_target_properties(distance_metrics PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")