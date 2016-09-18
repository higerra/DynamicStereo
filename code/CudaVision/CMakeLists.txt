cmake_minimum_required(VERSION 3.0)

project(CudaVision)

find_package(CUDA REQUIRED)

#test
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/CudaVision)
include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_executable(CudaStereoTest cuda_stereo_test.cpp cuStereoUtil.h ${ROOT_DIRECTORY}/test/test_main.cpp)
target_link_libraries(CudaStereoTest gtest ${OpenCV_LIBS} ${THEIA_LIBRARIES} glog)
add_test(NAME CudaStereoTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CudaStereoTest)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_30,code=sm_30")
cuda_add_library(cudaStereoMatching cuCamera.h cuStereoUtil.h cuStereoMatching.h cudaStereoMatching.cu)