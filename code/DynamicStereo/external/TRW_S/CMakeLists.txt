cmake_minimum_required(VERSION 3.0)

project(TRW_S)

file(GLOB SOURCE_FILES "*.cpp")
file(GLOB HEADER_FILES "*.h")

add_library(TRW_S ${SOURCE_FILES} ${HEADER_FILES})