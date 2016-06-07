cmake_minimum_required(VERSION 3.0)

project(Contour)

set(SOURCE_FILES contourdev_cli.cpp contourdev.cpp)
set(HEADER_FILES contourdev.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/contour_dev)

include_directories(/usr/local/include/eigen3 ${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation)
add_executable(Contour ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(Contour base gflags glog common segment_gb video_segment_util)
