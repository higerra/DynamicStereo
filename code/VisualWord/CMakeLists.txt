cmake_minimum_required(VERSION 3.0)

project(VisualWord)

find_package(OpenCV REQUIRED)

set(SOURCE_FILE visualword_cli.cpp visualword.cpp)
set(HEADER_FILE visualword.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VisualWord)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation /usr/local/include/eigen3)

add_executable(VisualWord ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(VisualWord ${OpenCV_LIBS} common video_segment_util gflags glog)