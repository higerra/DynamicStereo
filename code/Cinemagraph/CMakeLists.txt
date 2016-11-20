cmake_minimum_required(VERSION 3.0)

project(Cinemagraph)

find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/Cinemagraph)

include_directories(/usr/local/include/eigen3)
add_library(CinemagraphModule cinemagraph.h cinemagraph.cpp)
add_library(CinemagraphUtility cinemagraph_util.h cinemagraph_util.cpp)

target_link_libraries(CinemagraphModule ${OpenCV_LIBS} glog)
target_link_libraries(CinemagraphUtility CinemagraphModule ${CERES_LIBRARIES})

add_executable(Cinemagraph cinemagraph_cli.cpp)
target_link_libraries(Cinemagraph CinemagraphModule CinemagraphUtility gflags)