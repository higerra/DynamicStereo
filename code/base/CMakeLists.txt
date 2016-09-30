cmake_minimum_required(VERSION 3.0)

project(base)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(HEADER_FILE camera_utli.h superpixel.h colorwheel.h opticalflow.h SLIC.h
#        thread_guard.h configurator.h file_io.h utility.h plane3D.h)
#set(SOURCE_FILE  camera_utli.cpp depth.cpp superpixel.cpp colorwheel.cpp opticalflow.cpp SLIC.cpp
#        configurator.cpp utility.cpp plane3D.cpp)

set(HEADER_FILE depth.h thread_guard.h file_io.h utility.h)
set(SOURCE_FILE  depth.cpp utility.cpp)

include_directories(/usr/local/include/eigen3)
#add_subdirectory(OpticalFlow)
add_library(base ${HEADER_FILE} ${SOURCE_FILE})

target_link_libraries(base opencv_core opencv_imgcodecs)
#target_link_libraries(base OpticalFlow)
target_link_libraries(base glog)