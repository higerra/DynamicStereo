cmake_minimum_required(VERSION 3.0)

project(segment_gb)

add_library(segment_gb segment-image.h segment-image.cpp disjoint-set.h disjoint-set.cpp segment-graph.h segment-graph.cpp)

target_link_libraries(segment_gb ${OpenCV_LIBS} glog)