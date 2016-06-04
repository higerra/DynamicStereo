cmake_minimum_required(VERSION 3.0)

project(SegmentAndRender)

set(SOURCE_FILES dynamicregularizer.cpp dynamicsegment_display.cpp dynamicsegment_flashy.cpp
        segment_render_cli.cpp mfgrabcut2.cpp)
set(HEADER_FILES dynamicsegment.h dynamicregularizer.h gcgraph.hpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SegmentAndRender)

include_directories(/usr/local/include/eigen3 ${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation)
add_executable(SegmentAndRender ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(SegmentAndRender base gflags glog common segment_gb video_segment_util MRF)
