cmake_minimum_required(VERSION 3.0)

project(DynamicStereo)


set(SOURCE_FILES dynamicstereo.cpp dynamicstereo_cli.cpp MRFDefinition.cpp proposal.cpp optimization_firstorder.cpp optimization_fusionmove.cpp optimization_TRWS.cpp
        local_matcher.cpp)
set(HEADER_FILES dynamicstereo.h proposal.h optimization.h local_matcher.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/DynamicStereo)

add_subdirectory(external/segment_ms)
add_subdirectory(external/segment_gb)
add_subdirectory(external/QPBO1.4)
add_subdirectory(external/MRF2.2)
add_executable(DynamicStereo ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(DynamicStereo base gflags glog ${THEIA_LIBRARIES} segment_ms segment_gb MRF QPBO1.4)
