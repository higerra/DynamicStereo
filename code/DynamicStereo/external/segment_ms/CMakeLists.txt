cmake_minimum_required(VERSION 3.0)

project(segment_ms)

set(HEADER_FILES ms.h msImageProcessor.h msSys.h RAList.h rlist.h tdef.h)
set(SOURCE_FILES ms.cpp msImageProcessor.cpp msSys.cpp RAList.cpp rlist.cpp)

add_library(segment_ms ${SOURCE_FILES} ${HEADER_FILES})