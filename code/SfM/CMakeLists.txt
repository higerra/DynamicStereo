cmake_minimum_required(VERSION 3.0)

project(SfM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SfM)

include_directories(${THEIA_INCLUDE_DIRS} /usr/local/include/eigen3)

add_executable(SfM command_line_helpers.h SfM_cli.cpp)

target_link_libraries(SfM base glog gflags ${THEIA_LIBRARIES})
