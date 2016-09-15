cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(ROOT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
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
    add_subdirectory(${ROOT_DIRECTORY}/CudaVison)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#supporting code
add_subdirectory(${ROOT_DIRECTORY}/base)
add_subdirectory(${ROOT_DIRECTORY}/external/MRF2.2)
add_subdirectory(${ROOT_DIRECTORY}/external/segment_ms)
add_subdirectory(${ROOT_DIRECTORY}/external/segment_gb)
add_subdirectory(${ROOT_DIRECTORY}/external/line_util)
add_subdirectory(${ROOT_DIRECTORY}/external/video_segmentation)
#add_subdirectory(${ROOT_DIRECTORY}/external/KMeansRex)
add_subdirectory(${ROOT_DIRECTORY}/VideoSegmentation)
add_subdirectory(${ROOT_DIRECTORY}/MLModule)
add_subdirectory(${ROOT_DIRECTORY}/SubspaceStab)
#main code
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DataConvert)
add_subdirectory(${ROOT_DIRECTORY}/SfM)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Line)
add_subdirectory(${ROOT_DIRECTORY}/DynamicStereo)
add_subdirectory(${ROOT_DIRECTORY}/SegmentAndRender)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DynamicConfidence)
add_subdirectory(${ROOT_DIRECTORY}/test)
#add_subdirectory(${ROOT_DIRECTORY}/TrainData)
#add_subdirectory(${ROOT_DIRECTORY}/RandomForest)

add_subdirectory(${ROOT_DIRECTORY}/VisualWord)
#add_subdirectory(${ROOT_DIRECTORY}/extractFeature)
#add_subdirectory(${ROOT_DIRECTORY}/classifier)
#add_subdirectory(${ROOT_DIRECTORY}/contour_dev)
