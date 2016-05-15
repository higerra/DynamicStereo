cmake_minimum_required(VERSION 3.0)

project(tempMeanshift)

add_executable(tempMeanshift tempMeanshift_cli.cpp tempMeanshift.h tempMeanshift.cpp)

target_link_libraries(tempMeanshift gflag glog segment_ms base)