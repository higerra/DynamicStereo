cmake_minimum_required(VERSION 3.0)

project(DynamicConfidence)


set(SOURCE_FILES dynamic_confidence.cpp dynamic_confidence_cli.cpp)
set(HEADER_FILES dynamic_confidence.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DynamicConfidence)

add_executable(DynamicConfidence ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(DynamicConfidence base gflags glog ${THEIA_LIBRARIES})
