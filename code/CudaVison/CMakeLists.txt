cmake_minimum_required(VERSION 3.0)

project(CudaVision)

find_package(CUDA 8 REQUIRED)

add_library(CudaStereo cuda_stereo.h cuda_stereo.cpp)

#test
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/CudaVision)
include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_executable(CudaStereoTest cuda_stereo.cpp cuda_stereo_test.cpp ${ROOT_DIRECTORY}/test/test_main.cpp)
target_link_libraries(CudaStereoTest gtest ${THEIA_LIBRARIES})
add_test(NAME CudaStereoTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CudaStereoTest)