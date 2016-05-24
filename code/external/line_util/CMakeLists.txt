cmake_minimum_required(VERSION 3.0)

project(line_util)

include_directories(/usr/local/include/eigen3)

add_subdirectory(vpdetection)

add_library(line_util line_util.cpp line_util.h)
target_link_libraries(line_util VPdetection glog)