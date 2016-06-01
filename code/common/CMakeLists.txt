cmake_minimum_required(VERSION 3.0)

project(common)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/common)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_library(common descriptor.cpp descriptor.h
        regiondescriptor.h regiondescriptor.cpp
        dynamic_utility.h dynamic_utility.cpp
        dynamicwarpping.h dynamicwarpping.cpp
        stereomodel.h)
target_link_libraries(common line_util segment_ms segment_gb)

target_link_libraries(common base glog ${THEIA_LIBRARIES} OpenMeshCore)
