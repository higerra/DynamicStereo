cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)

set(SOURCE_FILES dynamicstereo.cpp dynamicstereo_cli.cpp MRFDefinition.cpp
        local_matcher.cpp optimization_firstorder.cpp)
set(HEADER_FILES dynamicstereo.h optimization.h local_matcher.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DynamicStereo)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_executable(DynamicStereo ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(DynamicStereo base gflags glog ${THEIA_LIBRARIES} MRF GeometryModule)
target_link_libraries(DynamicStereo opencv_imgproc opencv_videoio)

if(CUDA_FOUND)
	set(NVCC_CXX_FLAGS "${NVCC_CXX_FLAGS} -std=c++11 -O3 -gencode arch=compute_50,code=sm_50")
    cuda_add_library(cudaWrapper cudaWrapper.h cudaWrapper.cu)
    target_link_libraries(DynamicStereo cudaWrapper)
endif()

