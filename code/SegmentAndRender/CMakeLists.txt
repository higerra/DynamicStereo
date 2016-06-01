cmake_minimum_required(VERSION 3.0)

project(SegmentAndRender)

set(SOURCE_FILES dynamicregularizer.cpp dynamicsegment_display.cpp dynamicsegment_flashy.cpp dynamicwarpping.cpp
        segment_render_cli.cpp)
set(HEADER_FILES dynamicsegment.h dynamicregularizer.h dynamicwarpping.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SegmentAndRender)

include_directories(/usr/local/include/eigen3)
add_executable(SegmentAndRender ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(SegmentAndRender base gflags glog MRF video_segment_util)
