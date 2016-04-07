cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/test)

include_directories(/usr/local/include/eigen3)
add_subdirectory(googletest)
add_executable(utilityTest test_main.cpp utility_test.cpp)
target_link_libraries(utilityTest base gtest)

add_test(NAME utilityTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utilityTest)