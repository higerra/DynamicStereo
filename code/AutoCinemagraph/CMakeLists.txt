cmake_minimum_required(VERSION 3.0)

project(AutoCinemagraph)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_BUILD_TYPE Relase)

add_executable(AutoCinemagraph auto_cinemagraph.h auto_cinemagraph.cpp auto_cinemagraph_cli.cpp)

target_link_libraries(AutoCinemagraph opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio
        opencv_cudacodec opencv_cudaarithm opencv_cudaimgproc opencv_cudafeatures2d opencv_cudafilters opencv_cudalegacy
        opencv_cudaobjdetect opencv_cudawarping opencv_cudev
        opencv_cudaoptflow gflags glog)

