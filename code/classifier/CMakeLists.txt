cmake_minimum_required(VERSION 3.0)

project(classifier)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/classifier)

add_executable(classifier classifier_cli.cpp classifier.h classifier.cpp)

target_link_libraries(classifier gflags glog base)