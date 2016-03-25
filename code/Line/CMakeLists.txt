cmake_minimum_required(VERSION 3.0)

project(Line)

set(HEADER_FILES lineSeg.h)
set(SOURCE_FILES lineSeg.cpp line_cli.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/Line)

add_executable(Line ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(Line ${THEIA_LIBRARIES} ${OPENCV_LIBRARIES} gflags glog base)