cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/test)

include_directories(/usr/local/include/eigen3)
add_subdirectory(googletest)

add_executable(utilityTest test_main.cpp utility_test.cpp)
target_link_libraries(utilityTest base gtest gflags)
add_test(NAME utilityTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utilityTest)

#add_executable(classifierTest ${ROOT_DIRECTORY}/classifier/classifier_test.cpp ${ROOT_DIRECTORY}/classifier/classifier.cpp test_main.cpp)
#target_link_libraries(classifierTest GeometryModule gtest base)
#add_test(NAME classifierTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/classifierTest)
