cmake_minimum_required(VERSION 3.0)

project(VideoSegmentation)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(GMM_LIB colorGMM.h colorGMM.cpp)
add_library(ColorGMM ${GMM_LIB})
target_link_libraries(ColorGMM glog ${OpenCV_LIB})

set(VIDEO_SEGMENT_LIB pixel_feature.h distance_metric.h pixel_feature.cpp
        videosegmentation.h videosegmentation.cpp mfgrabcut2.cpp)

include_directories(/usr/local/include/eigen3)
add_library(VideoSegmentModule ${VIDEO_SEGMENT_LIB})
target_link_libraries(VideoSegmentModule ColorGMM segment_gb MRF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VideoSegmentation)

add_executable(VideoSegmentation videosegmentation_cli.cpp)
target_link_libraries(VideoSegmentation gflags VideoSegmentModule)

#test
add_executable(VideoSegmentation_test ${ROOT_DIRECTORY}/test/test_main.cpp videosegmentation_test.cpp)
target_link_libraries(VideoSegmentation_test VideoSegmentModule gtest)
add_test(NAME VideoSegmentation_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/VideoSegmentation_test)