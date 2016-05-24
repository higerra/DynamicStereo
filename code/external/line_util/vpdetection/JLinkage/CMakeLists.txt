cmake_minimum_required(VERSION 3.0)

project(JLinkage)

set(CMAKE_BUILD_TYPE Release)

include_directories(include)

set(SOURCE_FILES src/JLinkage.cpp src/Utilities.cpp src/RandomSampler.cpp)

add_library(JLinkage ${SOURCE_FILES})
