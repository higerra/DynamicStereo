#-------------------------------------------------
#
# Project created by QtCreator 2015-09-23T15:57:13
#
#-------------------------------------------------

QT       += core gui widgets opengl

TARGET = Viewer
CONFIG   += console
CONFIG   -= app_bundle
CONFIG += c++11

TEMPLATE = app


SOURCES += main.cpp \
    main_widget.cpp \
    ../base/depth.cpp \
    ../base/utility.cpp\
    navigation.cpp \
    videorenderer.cpp \
    scene.cpp \
    framerecorder.cpp

HEADERS += \
    main_widget.h \
    ../base/depth.h \
    ../base/file_io.h \
    ../base/utility.h\
    navigation.h \
    animation.h \
    videorenderer.h \
    scene.h \
    framerecorder.h


INCLUDEPATH += '/usr/local/include/eigen3'
INCLUDEPATH += '/usr/local/include'

LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -ltheia

unix:!macx{
    LIBS += -L/usr/lib/x86_64-linux-gnu/ -lGLU
}

RESOURCES += \
    shader.qrc

DISTFILES += \
    blend.vert \
    blend.frag \
    video_blend.vert \
    video_blend.frag
