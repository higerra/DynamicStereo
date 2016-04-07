cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(CMAKE_BUILD_TYPE Release)
set(BUILD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(Theia REQUIRED)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/base)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/MRF2.2)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DataConvert)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SfM)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Line)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DynamicStereo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DynamicConfidence)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)