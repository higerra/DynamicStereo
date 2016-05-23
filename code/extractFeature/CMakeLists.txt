cmake_minimum_required(VERSION 3.0)

project(extractFeature)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/extractFeature)

include_directories(/usr/local/include/eigen3)

add_library(featureModule descriptor.h descriptor.cpp)
target_link_libraries(featureModule segment_ms)

add_executable(extractFeature extractfeature_cli.cpp extracfeature.h extractfeature.cpp)

target_link_libraries(extractFeature gflags glog featureModule base)