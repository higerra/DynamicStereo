cmake_minimum_required(VERSION 3.0)

project(common)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/common)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_library(common descriptor.cpp descriptor.h
        regiondescriptor.h regiondescriptor.cpp
        dynamic_utility.h dynamic_utility.cpp
        dynamicwarpping.h dynamicwarpping.cpp
        CVdescriptor.h CVdescriptor.cpp
        stereomodel.h)

target_link_libraries(common segment_ms segment_gb)
target_link_libraries(common base glog ${THEIA_LIBRARIES} OpenMeshCore video_segment_lib)

add_executable(common_test common_test.cpp ${ROOT_DIRECTORY}/test/test_main.cpp)
target_link_libraries(common_test common gtest)
add_test(COMMAND common_test NAME common_test)

