cmake_minimum_required(VERSION 3.0)

project(QPBO1.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(HEADER_FILES block.h QPBO.h ELC.h ELC0.h)
set(SOURCE_FILES QPBO.cpp QPBO_extra.cpp QPBO_maxflow.cpp QPBO_postprocessing.cpp)

add_library(QPBO1.4 ${HEADER_FILES} ${SOURCE_FILES})