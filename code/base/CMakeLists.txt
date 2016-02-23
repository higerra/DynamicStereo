cmake_minimum_required(VERSION 3.0)

project(base)

set(HEADER_FILE camera.h camera_utli.h frame.h superpixel.h colorwheel.h opticalflow.h SLIC.h thread_guard.h configurator.h file_io.h meanshift.h utility.h)
set(SOURCE_FILE camera.cpp camera_utli.cpp depth.cpp frame.cpp superpixel.cpp colorwheel.cpp opticalflow.cpp SLIC.cpp configurator.cpp meanshift.cpp utility.cpp)

add_subdirectory(OpticalFlow)
add_library(base ${HEADER_FILE} ${SOURCE_FILE})

unset(OPENMP_FOUND)
unset(CUDA_FOUND)
find_package(OpenMP QUIET)
find_package(CUDA QUIET)
if(OPENMP_FOUND)
    message("-- Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("-- Compiler flags: ${CMAKE_CXX_FLAGS}")
endif()

if(CUDA_FOUND)
    message("-- Found CUDA")
    add_definitions("-DUSE_CUDA")
endif()



target_link_libraries(base ${OpenCV_LIBS})
target_link_libraries(base OpticalFlow)
target_link_libraries(base glog)