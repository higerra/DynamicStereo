cmake_minimum_required(VERSION 3.0)

project(VisualWord)

find_package(OpenCV REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VisualWord)

include_directories(/usr/local/include/eigen3)

add_library(VisualWordModule visualword.cpp visualword.h)
target_link_libraries(VisualWordModule VideoSegmentModule MLModule glog)

add_executable(VisualWord visualword_cli.cpp)

target_link_libraries(VisualWord ${OpenCV_LIBS} VisualWordModule gflags glog)