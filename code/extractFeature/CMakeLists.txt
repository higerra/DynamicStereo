cmake_minimum_required(VERSION 3.0)

project(extractFeature)

add_executable(extractFeature extractfeature_cli.cpp extracfeature.h extractfeature.cpp)

target_link_libraries(extractFeature gflag glog segment_ms base)