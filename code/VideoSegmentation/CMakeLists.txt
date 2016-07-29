cmake_minimum_required(VERSION 3.0)

project(VideoSegmentation)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(VIDEO_SEGMENT_LIB pixel_feature.h distance_metric.h pixel_feature.cpp)

add_library(video_segment_lib ${VIDEO_SEGMENT_LIB})

set(SOURCE_FILE videosegmentation.cpp videosegmentation_cli.cpp)
set(HEADER_FILE videosegmentation.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VideoSegmentation)

add_executable(VideoSegmentation ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(VideoSegmentation ${OpenCV_LIBS} gflags glog common segment_gb video_segment_lib)