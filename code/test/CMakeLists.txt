cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/test)

include_directories(/usr/local/include/eigen3)
add_subdirectory(googletest)

add_executable(utilityTest test_main.cpp utility_test.cpp)
target_link_libraries(utilityTest base gtest)

add_executable(descriptorTest ${ROOT_DIRECTORY}/extractFeature/descriptor_test.cpp test_main.cpp)
target_link_libraries(descriptorTest featureModule gtest base)

add_test(NAME utilityTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utilityTest)
add_test(NAME descriptorTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}descriptorTest)