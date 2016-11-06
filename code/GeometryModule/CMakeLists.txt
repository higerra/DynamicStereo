cmake_minimum_required(VERSION 3.0)

project(GeometryModule)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_library(GeometryModule dynamic_utility.h dynamic_utility.cpp
        dynamicwarpping.h dynamicwarpping.cpp
        sfmmodel.h sfmmodel.cpp
        stereomodel.h)

target_link_libraries(GeometryModule base glog ${THEIA_LIBRARIES} OpenMeshCore OpenMeshTools)
