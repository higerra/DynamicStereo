cmake_minimum_required(VERSION 3.0)
find_package(GTest REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/test)
include_directories(${THEIA_INDLUE_DIRS} /usr/local/include/eigen3)
add_executable(utilityTest test_main.cpp utility_test.cpp)
target_link_libraries(utilityTest base ${GTEST_LIBRARIES})

add_test(NAME utilityTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utilityTest)