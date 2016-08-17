cmake_minimum_required(VERSION 3.0)

project(VideoSegmentation)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

include_directories(/usr/local/include/eigen3)
######################################################
#GMM library
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
    #GPU version
    message("-- Compiling GPU version")
    set(GMM_LIB colorGMM.h colorGMM.cpp colorGMMcu.cu)
    cuda_add_library(ColorGMM ${GMM_LIB})
    target_link_libraries(ColorGMM glog ${OpenCV_LIBS} ${CUDA_LIBRARIES})
else(CUDA_FOUND)
    #CPU version
    set(GMM_LIB colorGMM.h colorGMM.cpp)
    message("-- Compiling CPU version")
    add_library(ColorGMM ${GMM_LIB})
    target_link_libraries(ColorGMM glog ${OpenCV_LIB})
endif(CUDA_FOUND)



######################################################3
#VideoSegmentation libraryx
set(VIDEO_SEGMENT_LIB pixel_feature.h distance_metric.h pixel_feature.cpp
        videosegmentation.h videosegmentation.cpp mfgrabcut2.cpp)

add_library(VideoSegmentModule ${VIDEO_SEGMENT_LIB})
target_link_libraries(VideoSegmentModule ColorGMM segment_gb MRF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/VideoSegmentation)

add_executable(VideoSegmentation videosegmentation_cli.cpp)
target_link_libraries(VideoSegmentation gflags VideoSegmentModule)

#test
add_executable(VideoSegmentation_test ${ROOT_DIRECTORY}/test/test_main.cpp videosegmentation_test.cpp)
target_link_libraries(VideoSegmentation_test VideoSegmentModule gtest)
add_test(NAME VideoSegmentation_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/VideoSegmentation_test)
