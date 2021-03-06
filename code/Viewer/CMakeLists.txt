cmake_minimum_required(VERSION 3.0)

project(Viewer)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/Viewer)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets OpenGL)

include_directories(/usr/local/include/eigen3 ${THEIA_INCLUDE_DIRS})
set(SOURCE_FILE main.cpp main_widget.cpp navigation.cpp scene.cpp videorenderer.cpp)
set(HEADER_FILE animation.h main_widget.h navigation.h scene.h videorenderer.h)
set(RESOURCE_FILE shader.qrc)
add_executable(Viewer ${SOURCE_FILE} ${HEADER_FILE} ${RESOURCE_FILE})

target_link_libraries(Viewer ${THEIA_LIBRARIES} ${OpenCV_LIBS} base GeometryModule CinemagraphModule
        Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Gui glog gflags GLU GL)