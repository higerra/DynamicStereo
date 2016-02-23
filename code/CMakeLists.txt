cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(CMAKE_BUILD_TYPE Release)
set(BUILD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

include_directories(/usr/local/include/eigen3 ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/base)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/MRF2.2)