cmake_minimum_required(VERSION 3.0)

project(VideoSegmentation)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(VIDEO_SEGMENT_LIB pixel_feature.h distance_metric.h pixel_feature.cpp)

add_library(video_segment_lib ${VIDEO_SEGMENT_LIB})
target_link_libraries(video_segment_lib ${OpenCV_LIBS} glog)

set(SOURCE_FILE videosegmentation.cpp videosegmentation_cli.cpp)
set(HEADER_FILE videosegmentation.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VideoSegmentation)

add_executable(VideoSegmentation ${SOURCE_FILE} ${HEADER_FILE})
target_link_libraries(VideoSegmentation gflags common segment_gb video_segment_lib)

#test
add_executable(VideoSegmentation_test ${ROOT_DIRECTORY}/test/test_main.cpp videosegmentation_test.cpp)
target_link_libraries(VideoSegmentation_test video_segment_lib gtest)
add_test(NAME VideoSegmentation_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/VideoSegmentation_test)