cmake_minimum_required(VERSION 3.0)
project(HoughForest)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILE CRForestDetector.cpp CRPatch.cpp CRTree.cpp HoG.cpp)
set(HEAER_FILE CRForest.h CRForestDetector.h CRPatch.h CRTree.h HoG.h)

add_library(HoughForest ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(HoughForest ${OpenCV_LIBS})
