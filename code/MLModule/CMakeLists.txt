cmake_minimum_required(VERSION 3.0)

project(MLModule)

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/MLModule)

include_directories(/usr/local/include/eigen3)

add_library(ML_lib regiondescriptor.h regiondescriptor.cpp
        mlutility.h mlutility.cpp
        type_def.h)

target_link_libraries(ML_lib ${OpenCV_lib} glog)