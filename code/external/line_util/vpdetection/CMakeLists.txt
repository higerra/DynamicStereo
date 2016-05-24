cmake_minimum_required(VERSION 3.0)

project(VPdetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wno-null-conversion -Wno-extra-tokens")
set(SOURCE_FILES VPCluster.cpp VPSample.cpp updator.cpp)
set(HEADER_FILES VPCluster.h VPSample.h updator.h)

include_directories(JLinkage/include)

add_subdirectory(JLinkage)
add_library(VPdetection ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(VPdetection JLinkage)
