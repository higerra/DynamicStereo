cmake_minimum_required(VERSION 3.0)

project(SfM)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/SfM)

find_package(Theia REQUIRED)

include_directories(${THEIA_INCLUDE_DIRS})

add_executable(SfM command_line_helpers.h SfM_cli.cpp)

target_link_libraries(SfM base glog gflags ${THEIA_LIBRARIES} MRF)
