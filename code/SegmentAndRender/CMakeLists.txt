cmake_minimum_required(VERSION 3.0)

project(SegmentAndRender)

set(SOURCE_FILES dynamicregularizer.cpp dynamicsegment_display.cpp dynamicsegment_flashy.cpp
        segment_render_cli.cpp stabilization.cpp)
set(HEADER_FILES dynamicsegment.h dynamicregularizer.h stabilization.h stab_energy.h)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SegmentAndRender)

add_library(RPCA RPCA.h RPCA.cpp)

include_directories(/usr/local/include/eigen3 /usr/include/suitesparse ${CMAKE_CURRENT_BINARY_DIR}/../external/video_segmentation)
#link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)
add_executable(SegmentAndRender ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(SegmentAndRender base gflags glog common RPCA VideoSegmentModule VisualWordModule MRF cholmod spqr ${CERES_LIBRARIES})

#add test
add_executable(SegmentAndRender_test ${ROOT_DIRECTORY}/test/test_main.cpp render_test.cpp)
target_link_libraries(SegmentAndRender_test RPCA gtest glog)
add_test(NAME SegmentAndRender_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/SegmentAndRender_test)
