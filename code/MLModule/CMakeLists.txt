cmake_minimum_required(VERSION 3.0)

project(MLModule)

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/MLModule)

include_directories(/usr/local/include/eigen3)

add_library(MLModule regiondescriptor.h regiondescriptor.cpp
        CVdescriptor.h CVdescriptor.cpp
        mlutility.h mlutility.cpp
        type_def.h)

target_link_libraries(MLModule opencv_core opencv_ml opencv_features2d opencv_xfeatures2d glog)