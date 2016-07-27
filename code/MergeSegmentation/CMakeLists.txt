cmake_minimum_required(VERSION 3.0)

project(MergeSegmentation)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(SOURCE_FILE mergesegmentation.cpp mergesegmentation_cli.cpp)
set(HEADER_FILE mergesegmentation.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/MergeSegmentation)

add_executable(MergeSegmentation ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(MergeSegmentation ${OpenCV_LIBS} gflags glog common segment_gb)