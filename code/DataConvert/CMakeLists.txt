cmake_minimum_required(VERSION 3.0)

project(DataConvert)

set(SOURCE_FILES dataconvert_cli.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DataConvert)
add_executable(DataConvert ${SOURCE_FILES})

target_link_libraries(DataConvert base ${THEIA_LIBRARIES} gflags glog)