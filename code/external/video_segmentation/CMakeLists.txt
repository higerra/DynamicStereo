cmake_minimum_required(VERSION 3.0)

project(video_segment_reader)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(/usr/local/include /usr/include ${CMAKE_CURRENT_BINARY_DIR})

link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vs_base)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/segment_util)
