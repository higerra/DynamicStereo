cmake_minimum_required(VERSION 3.0)

project(KMeansRex)

set(SOURCE_FILE KMeansRexCore.cpp)
set(HEADER_FILE KMeansRexCore.h mersenneTwister2002.c)

include_directories(/usr/local/include/eigen3)

add_library(KMeansRex ${SOURCE_FILE} ${HEADER_FILE})