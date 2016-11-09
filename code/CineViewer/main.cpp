#include <QApplication>
#include <QLabel>
#include <QBoxLayout>
#include <iostream>

#include "../base/file_io.h"
#ifndef QT_NO_OPENGL
#include <main_widget.h>
#endif

using namespace std;
using namespace dynamic_stereo;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setApplicationName("Dynamic Renderer");
    if(argc < 2){
        cerr << "Usage: ./Viewer <path-to-directory>"<<endl;
        exit(-1);
    }
#ifndef QT_NO_OPENGL
    QWidget window;
    window.resize(1280, 720);

    QGLFormat fmt;
    //fmt.setVersion(3,3);
    //fmt.setProfile(QGLFormat::CompatibilityProfile);
    //fmt.setDoubleBuffer(true);
    //fmt.setAlpha(true);
    MainWidget *widget = new MainWidget(string(argv[1]), fmt);
    QHBoxLayout * layout = new QHBoxLayout();

    layout->addWidget(widget);
    window.setLayout(layout);
    window.show();
#else
    QLabel note("OpenGL support required");
    note.show();
#endif
    return a.exec();
}
