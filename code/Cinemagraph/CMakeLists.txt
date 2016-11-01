cmake_minimum_required(VERSION 3.0)

project(Cinemagraph)

find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY '${BUILD_DIRECTORY}/Cinemagraph)

include_directories(/usr/local/include/eigen3)
add_library(CinemagraphModule cinemagraph.h cinemagraph.cpp)

target_link_libraries(CinemagraphModule opencv_core opencv_highgui opencv_imgcodecs opencv_videoio glog)

add_executable(Cinemagraph cinemagraph_cli.cpp)
target_link_libraries(Cinemagraph CinemagraphModule gflags)