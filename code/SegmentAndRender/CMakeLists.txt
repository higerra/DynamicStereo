cmake_minimum_required(VERSION 3.0)

project(SegmentAndRender)

set(SOURCE_FILES dynamicregularizer.cpp dynamicsegment_display.cpp dynamicsegment_flashy.cpp RPCA.cpp
        segment_render_cli.cpp)
set(HEADER_FILES dynamicsegment.h dynamicregularizer.h RPCA.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SegmentAndRender)

include_directories(/usr/local/include/eigen3 /usr/include/suitesparse ${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation)
#link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)
add_executable(SegmentAndRender ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(SegmentAndRender base gflags glog common VideoSegmentModule VisualWordModule MRF cholmod spqr)
