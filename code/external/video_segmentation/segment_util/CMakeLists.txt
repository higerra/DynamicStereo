cmake_minimum_required(VERSION 2.6)

project(video_segment_util)

set(SOURCES segmentation_io.cpp
            segmentation_util.cpp
            segmentation_render.cpp
            segmentation_boundary.cpp
            )
set(HEADERS segmentation_io.h
        segmentation_util.h
        segmentation_render.h
        segmentation_boundary.h
        )

set(SOURCES "${SOURCES}" "${HEADERS}")

find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HDR segmentation.proto)

add_library(video_segment_util ${SOURCES} ${PROTO_SRC} ${PROTO_HDR})
target_link_libraries(video_segment_util ${OpenCV_LIBS} glog video_segment_base)
