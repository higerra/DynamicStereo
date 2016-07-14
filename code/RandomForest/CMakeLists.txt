cmake_minimum_required(VERSION 3.0)

project(RandomForest)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(SROUCE_FILE randomforest_cli.cpp randomforest.cpp)
set(HEADER_FILE randomforest.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/RandomForest)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation /usr/local/include/eigen3)

add_executable(RandomForest ${SROUCE_FILE} ${HEADER_FILE})

target_link_libraries(RandomForest ${OpenCV_LIBS} video_segment_util common gflags glog)