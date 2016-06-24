cmake_minimum_required(VERSION 3.0)
project(TrainData)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/TrainData)

set(SOURCE_FILE traindata_cli.cpp traindata.cpp)
set(HEADER_FILE traindata.h)

add_executable(TrainData ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(TrainData glog gflags ${OpenCV_LIBS})