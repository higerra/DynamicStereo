cmake_minimum_required(VERSION 2.6)

project(video_segment_base)

set(SOURCES base.cpp
            base_impl.cpp)
set(HEADER base.h
        base_impl.h)
set(SOURCES "${SOURCES}" "${HEADERS}")

add_library(video_segment_base ${SOURCES})
target_link_libraries(video_segment_base glog)