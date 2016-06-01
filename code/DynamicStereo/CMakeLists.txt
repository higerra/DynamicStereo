cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)

set(SOURCE_FILES dynamicstereo.cpp dynamicstereo_cli.cpp MRFDefinition.cpp warpping.cpp
        local_matcher.cpp optimization_firstorder.cpp)
set(HEADER_FILES dynamicstereo.h optimization.h local_matcher.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DynamicStereo)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
add_executable(DynamicStereo ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(DynamicStereo base gflags glog ${THEIA_LIBRARIES} MRF common)
