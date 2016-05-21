cmake_minimum_required(VERSION 3.0)

project(base)

set(HEADER_FILE frame.h camera.h camera_utli.h superpixel.h colorwheel.h opticalflow.h SLIC.h
        thread_guard.h configurator.h file_io.h utility.h plane3D.h)
set(SOURCE_FILE frame.cpp camera.cpp camera_utli.cpp depth.cpp superpixel.cpp colorwheel.cpp opticalflow.cpp SLIC.cpp
        configurator.cpp utility.cpp plane3D.cpp)

include_directories(/usr/local/include/eigen3)
add_subdirectory(OpticalFlow)
add_library(base ${HEADER_FILE} ${SOURCE_FILE})

target_link_libraries(base ${OpenCV_LIBS})
target_link_libraries(base OpticalFlow)
target_link_libraries(base glog)