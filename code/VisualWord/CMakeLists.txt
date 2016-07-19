cmake_minimum_required(VERSION 3.0)

project(VisualWord)

find_package(OpenCV REQUIRED)

set(SOURCE_FILE visualword_cli.cpp visualword.cpp)
set(HEADER_FILE visualword.h)

add_executable(VisualWord ${SOURCE_FILE} ${HEADER_FILE})

target_link_libraries(VisualWord ${OpenCV_LIBS} common)