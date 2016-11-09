QT   += core gui widgets opengl


CONFIG += c++11

TARGET = CineViewer
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    main_widget.cpp \
    ../base/depth.cpp \
    ../base/utility.cpp\
    ../GeometryModule/stereomodel.cpp\
    navigation.cpp \
    scene.cpp\

HEADERS += \
    main_widget.h \
    ../base/depth.h \
    ../base/file_io.h \
    ../base/utility.h\
    ../GeometryModule/stereomodel.h\
    navigation.h \
    animation.h \
    scene.h\

INCLUDEPATH += '/usr/local/include/eigen3'
INCLUDEPATH += '/usr/include/suitesparse'
INCLUDEPATH += '/usr/local/include/theia/libraries/'
INCLUDEPATH += '/usr/local/include/theia/libraries/statx'
INCLUDEPATH += '/usr/local/include/theia/libraries/optimo'
INCLUDEPATH += '/usr/local/include/theia/libraries/vlfeat'
INCLUDEPATH += '/usr/local/include/theia/libraries/cereal'
INCLUDEPATH += '/usr/local/include/theia/libraries/visual_sfm'

INCLUDEPATH += '/usr/local/include'

LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -ltheia

unix:!macx{
    LIBS += -L/usr/lib/x86_64-linux-gnu/ -lGLU
    LIBS += -L/usr/lib/x86_64-linux-gnu/ -lglog
    LIBS += -L/usr/lib/x86_64-linux-gnu/ -lgflags
}

RESOURCES += \
    shader.qrc

DISTFILES += \
    blend.vert \
    blend.frag \
    video_blend.vert \
    video_blend.frag

