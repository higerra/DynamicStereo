cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)

set(SOURCE_FILES dynamicstereo.cpp dynamicstereo_cli.cpp MRFDefinition.cpp)
set(HEADER_FILES dynamicstereo.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DynamicStereo)

add_executable(DynamicStereo ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(DynamicStereo gflags glog ${THEIA_LIBRARIES} MRF base)