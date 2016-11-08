#ifndef MAIN_WIDGET_H
#define MAIN_WIDGET_H

#include <QGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShader>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QRectF>
#include <QImage>
#include <QMatrix4x4>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QTimerEvent>
#include <QBasicTimer>
#include <QFileDialog>
#include <QImage>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QMenuBar>
#include <QPoint>
#include <QContextMenuEvent>
#include <vector>
#include "../base/frame.h"
#include "../base/file_io.h"
#include "navigation.h"
#include "videorenderer.h"
#include "framerecorder.h"
#include "scene.h"
#include <memory>

namespace dynamic_stereo{

class MainWidget: public QGLWidget, protected QOpenGLFunctions{
    Q_OBJECT
public:
    explicit MainWidget(const std::string& data_directory,
                        const QGLFormat& fmt,
                         QWidget *parent = 0);
    ~MainWidget();
public slots:
    void switch_to_external(int);
    void switch_to_internal();
    void shut_down_display();

protected:
    void keyPressEvent(QKeyEvent *e) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void mouseDoubleClickEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void timerEvent(QTimerEvent *event) Q_DECL_OVERRIDE;
    void contextMenuEvent(QContextMenuEvent* e) Q_DECL_OVERRIDE;

private:
    void blendFrames(float percent);
    void open_external_video();

    static const int kNumBlendBuffers;
    const Eigen::Vector3f background;
    std::string path;
    int kNumScene;

    std::shared_ptr<QOpenGLShaderProgram> blend_program;

    GLuint blendframebuffer[2];
    GLuint blendtexture[2];
    GLuint renderbuffer_depth[2];

    GLfloat blendvertex_data[12];
    GLuint blendvertexBuffer;
    GLfloat blendtexcoord_data[8];
    GLuint blendtexcoordBuffer;
    GLuint blendindex_data[4];
    GLuint blendindexBuffer;

    std::vector<std::shared_ptr<Scene> > scenes;
    int imgwidth, imgheight;
    bool render_video;

    int mousex, mousey;
    int contextx, contexty;
    bool track_mouse;

    Navigation navigation;

    std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > >external_textures;
    //For animation
    QBasicTimer renderTimer;

    std::shared_ptr<QAction> action_internal;
    std::shared_ptr<QAction> action_shut;

    std::shared_ptr<FrameRecorder> frame_recorder;
    bool is_recording;

    void InitializeShader();

    void allocateResource();
    void freeResource();

};
} //namespace dynamic_stereo

#endif // MAIN_WIDGET_H
