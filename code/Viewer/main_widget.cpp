#include "main_widget.h"
#include "animation.h"
#include <fstream>
#include <iostream>

#ifdef __linux__
#include <GL/glu.h>
#else
#include <Opengl/glu.h>
#endif

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{

    const int MainWidget::kNumBlendBuffers = 2;

    MainWidget::MainWidget(const string& data_directory,const QGLFormat& fmt, QWidget *parent):
            QGLWidget(fmt,parent),
            path(data_directory),
            background(Vector3f(0,0,0)),
            track_mouse(true),
            navigation(data_directory),
            is_recording(false)
    {
        LOG(INFO) << "Initializing";
        scenes.resize(navigation.getNumFrames());
//    frame_recorder = shared_ptr<FrameRecorder>(new FrameRecorder(path));
        setFocusPolicy(Qt::StrongFocus);
        setMouseTracking(true);

        //Init gui
        action_shut = shared_ptr<QAction>(new QAction(tr("Shut Down"), this));
        action_internal = shared_ptr<QAction>(new QAction(tr("Video from dataset"), this));
        connect(action_internal.get(), SIGNAL(triggered()), this, SLOT(switch_to_internal()));
        connect(action_shut.get(), SIGNAL(triggered()), this, SLOT(shut_down_display()));
    }

    MainWidget::~MainWidget(){
        freeResource();
//    frame_recorder->exit();
    }



    void MainWidget::allocateResource(){
        glGenFramebuffers(kNumBlendBuffers, blendframebuffer);
        glGenTextures(kNumBlendBuffers, blendtexture);
        glGenRenderbuffers(kNumBlendBuffers, renderbuffer_depth);

        glEnable(GL_TEXTURE_2D);
        for(int i=0; i<kNumBlendBuffers; ++i){
            glBindTexture(GL_TEXTURE_2D, blendtexture[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width(), height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);

            glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer_depth[i]);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width(), height());

            glBindFramebuffer(GL_FRAMEBUFFER, blendframebuffer[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blendtexture[i], 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer_depth[i]);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if(status != GL_FRAMEBUFFER_COMPLETE){
                cerr << "Framebuffer not complete!" << endl;
                switch(status){
                    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                        cerr << "Incompete attachment" << endl;
                        break;
                    case GL_FRAMEBUFFER_UNSUPPORTED:
                        cerr << "Frame buffer unsupported" << endl;
                        break;
                    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                        cerr << "Missing attachment" << endl;
                        break;
                    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                        cerr << "Incomplete draw buffer" << endl;
                        break;
                    default:
                        cerr << "Unknown error" << endl;
                        break;
                }

                //exit(-1);
            }
        }
        glDisable(GL_TEXTURE_2D);

//    {
//        GLfloat* v = blendvertex_data;
//        v[0] = 0.0; v[1] = 0.0; v[2] = 0.0;
//        v[3] = (float)width(); v[4] = 0.0; v[5] = 0.0;
//        v[6] = (float)width(); v[7] = (float)height(); v[8] = 0.0;
//        v[9] = 0.0; v[10] = (float)height(); v[11] = 0.0;
//    }

//    {
//        GLfloat* t = blendtexcoord_data;
//        t[0] = 0.0; t[1] = 0.0;
//        t[2] = 1.0; t[3] = 0.0;
//        t[4] = 1.0; t[5] = 1.0;
//        t[6] = 0.0; t[7] = 1.0;
//    }

//    {
//        GLuint* bi = blendindex_data;
//        bi[0] = 0; bi[1] = 1; bi[2] = 2; bi[3] = 3;
//    }

//    glGenBuffers(1, &blendvertexBuffer);
//    glBindBuffer(GL_ARRAY_BUFFER, blendvertexBuffer);
//    glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), blendvertex_data, GL_STATIC_DRAW);

//    glGenBuffers(1, &blendtexcoordBuffer);
//    glBindBuffer(GL_ARRAY_BUFFER, blendtexcoordBuffer);
//    glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), blendtexcoord_data, GL_STATIC_DRAW);

//    glGenBuffers(1, &blendindexBuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, blendindexBuffer);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*sizeof(GLuint), blendindex_data, GL_STATIC_DRAW);

        const vector<int>& frame_ids = navigation.GetFrameIdx();
        for(int i=0; i<frame_ids.size(); i++){
            printf("Initializing scene %d\n", i);
            scenes[i].reset(new Scene(external_textures));
            if(!scenes[i]->initialize(path, frame_ids[i], navigation)){
                cerr << "Initalizing scene "<<i<<" failed"<<endl;
            }
        }
    }

    void MainWidget::freeResource(){

    }

    void MainWidget::InitializeShader(){
        blend_program = shared_ptr<QOpenGLShaderProgram>(new QOpenGLShaderProgram());
        if(!blend_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/blend.vert")){
            cerr << "Can not open blend vertex shader!" << endl;
            close();
        }
        if(!blend_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/blend.frag")){
            cerr << "Can not open blend fragment shader!" << endl;
            close();
        }
        if(!blend_program->link()){
            cerr << "Can not link blend shader!" << endl;
            close();
        }

        if(!blend_program->bind()){
            cerr << "Can not bind blend shader!" << endl;
            close();
        }
        blend_program->enableAttributeArray("vPosition");
        blend_program->enableAttributeArray("texcoord");
        blend_program->release();
    }

    void MainWidget::initializeGL(){
        initializeOpenGLFunctions();
        InitializeShader();

        glClearColor(background[0], background[1], background[2], 0.0);
        glEnable(GL_DEPTH);
        glClearDepth(0.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        allocateResource();
        //open_external_video();

        glViewport(0,0,width(), height());
        int blendNum = navigation.getBlendNum();
        renderTimer.start(1000.0/(float)blendNum, this);
    }

    void MainWidget::resizeGL(int w, int h){
        glViewport(0,0,w,h);
    }

    void MainWidget::open_external_video(){
        cout << "Opening external video" << endl;
        cv::Size dsize(400,400);
        const int border = 10;
        const int max_count = 300;
        int i = 0;
        char buffer[100] = {};
        while(true){
            sprintf(buffer, "external%03d.mp4", i++);
            cv::VideoCapture video;
            video.open(string(buffer));
            if(!video.isOpened()){
                cout << buffer << " not found" << endl;
                break;
            }
            cout << "Reading video:"<< buffer <<"..." << flush;
            vector<shared_ptr<QOpenGLTexture> > curtexture;
            int count = 0;
            while(true){
                if(count > max_count)
                    break;
                cv::Mat frame, small;
                if(!video.read(frame))
                    break;
                cv::resize(frame, small, cv::Size(dsize.width-2*border,dsize.height-2*border));
                cv::Mat teximg(dsize, CV_8UC3);
                for(int x=0; x<teximg.cols; ++x){
                    for(int y=0; y<teximg.rows; ++y){
                        if(x < border || y < border || x > teximg.cols-border || y > teximg.rows-border)
                            teximg.at<cv::Vec3b>(y,x) = cv::Vec3b(3,3,3);
                        else{
                            cv::Vec3b curpix = small.at<cv::Vec3b>(y-border, x-border);
                            uchar temp = curpix[0];
                            curpix[0] = curpix[2];
                            curpix[2] = temp;
                            if(curpix[0] < 3 && curpix[1] < 3 && curpix[2] < 3)
                                curpix = cv::Vec3b(3,3,3);
                            teximg.at<cv::Vec3b>(y,x) = curpix;
                        }
                    }
                }
                shared_ptr<QOpenGLTexture> curtex(new QOpenGLTexture(
                        QImage(teximg.data, teximg.cols, teximg.rows, QImage::Format_RGB888)));
                curtexture.push_back(curtex);
                count++;
            }
            external_textures.push_back(curtexture);
            cout<<"complete, frame count:" << count<<endl;
        }
    }

    void MainWidget::switch_to_internal(){
//    const pair<int,int>& f = navigation.getCurrentFrame();
//    scenes[f.first]->getVideoRenderer()
//            ->changeSource(f.second, contextx, contexty, videoRenderer::INTERNAL);
//    this->setFocus();
    }

    void MainWidget::switch_to_external(int id){
//    const pair<int,int>& f = navigation.getCurrentFrame();
//    scenes[f.first]->getVideoRenderer()
//            ->changeSource(f.second, contextx, contexty, videoRenderer::EXTERNAL,id);
//    this->setFocus();
    }

    void MainWidget::shut_down_display(){
//    const pair<int,int>& f = navigation.getCurrentFrame();
//    scenes[f.first]->getVideoRenderer()
//            ->changeSource(f.second, contextx, contexty, videoRenderer::STATIC);
//    this->setFocus();
    }

    void MainWidget::paintGL(){
        const int current_frame = navigation.getCurrentFrame();
        const int next_frame = navigation.getNextFrame();

        float percent = navigation.getFrameWeight();
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        if(navigation.getStatus() == Navigation::STATIC){
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            scenes[current_frame]->render(navigation);
        }else{
            glBindFramebuffer(GL_FRAMEBUFFER, blendframebuffer[0]);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            scenes[current_frame]->render(navigation);

            glBindFramebuffer(GL_FRAMEBUFFER, blendframebuffer[1]);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            scenes[next_frame]->render(navigation);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            blendFrames(percent);;
        }


//        glBindFramebuffer(GL_FRAMEBUFFER, 0);
//        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//        scenes[current_frame]->render(navigation);

//        glBindFramebuffer(GL_FRAMEBUFFER, blendframebuffer[0]);
//        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//        scenes[current_frame]->render(navigation);
//
//        glBindFramebuffer(GL_FRAMEBUFFER, blendframebuffer[1]);
//        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//        scenes[next_frame]->render(navigation);
//
//        glBindFramebuffer(GL_FRAMEBUFFER, 0);
//        blendFrames(percent);;


//    if(is_recording){
//        glReadBuffer(GL_FRONT);
//        shared_ptr<QImage> curimg(new QImage(width(), height(), QImage::Format_RGB888));
//        glReadPixels(0,0,width(), height(), GL_RGB, GL_UNSIGNED_BYTE, curimg->bits());
//        frame_recorder->submitFrame(curimg);
//    }
        glPopAttrib();
        glFlush();
    }

    void MainWidget::blendFrames(float percent){
        glPushAttrib(GL_ALL_ATTRIB_BITS);

        if(!blend_program->bind()){
            cerr << "Can not bind blend program" << endl;
            close();
        }
        glEnable(GL_TEXTURE_2D);
        QMatrix4x4 modelview, projection;
        modelview.setToIdentity();
        projection.setToIdentity();
        //projection.ortho(QRectF(0,0,navigation.GetFrameWidth(), navigation.GetFrameHeight()));
        projection.ortho(QRectF(0,0,width(), height()));


        blend_program->setUniformValue("mat_mv", modelview);
        blend_program->setUniformValue("mat_mp", projection);
        blend_program->setUniformValue("weight", percent);

        blend_program->setUniformValue("tex0",0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, blendtexture[0]);

        blend_program->setUniformValue("tex1",1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, blendtexture[1]);
        glEnable(GL_TEXTURE_2D);

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, width(), 0, height());

        glDisable(GL_DEPTH_TEST);
        glBegin(GL_QUADS);
        //glDisable(GL_DEPTH);
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);

        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(width(), 0.0f, 0.0f);

        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(width(), height(), 0.0f);

        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(0, height(), 0.0f);
        glEnd();

        glActiveTexture(GL_TEXTURE1);
        glDisable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glDisable(GL_TEXTURE_2D);

        blend_program->release();
        glPopAttrib();
    }

    void MainWidget::keyPressEvent(QKeyEvent *e){
        switch(e->key()){
            case(Qt::Key_P):{
                renderTimer.stop();
                break;
            }
            case(Qt::Key_S):{
                renderTimer.start(1000/(float)navigation.getBlendNum(), this);
                break;
            }
            case(Qt::Key_R):{
                is_recording = !is_recording;
                if(is_recording)
                    cout << "Recording on" << endl;
                else
                    cout << "Recording off" <<endl;
                break;
            }
            default:
                navigation.processKeyEvent(e);
                break;
        }
    }

    void MainWidget::mousePressEvent(QMouseEvent * e){
        track_mouse = false;
        mousex = e->x();
        mousey = e->y();
    }

    void MainWidget::mouseMoveEvent(QMouseEvent* e){
//    if(track_mouse){
//        mousex = e->x();
//        mousey = e->y();
//        const pair<int,int>& f = navigation.getCurrentFrame();
//        if(navigation.getStatus() == Navigation::STATIC)
//            scenes[f.first]->getVideoRenderer()
//                    ->setHighlight(f.second, mousex, mousey);
//    }
    }

    void MainWidget::mouseDoubleClickEvent(QMouseEvent *e){
        navigation.processMouseEvent(Navigation::DOUBLE_CLICK, 0, 0);
    }

    void MainWidget::mouseReleaseEvent(QMouseEvent* e){
        int dx = e->x() - mousex;
        int dy = e->y() - mousey;
        mousex = e->x();
        mousey = e->y();
        navigation.processMouseEvent(Navigation::MOVE, dx, dy);
        track_mouse = true;
    }

    void MainWidget::timerEvent(QTimerEvent* event){
        navigation.updateNavigation();
        updateGL();
    }

    void MainWidget::contextMenuEvent(QContextMenuEvent *e){
//    const pair<int,int>& f = navigation.getCurrentFrame();
//    const shared_ptr<videoRenderer>& vr = scenes[f.first]->getVideoRenderer();
//    if(vr->getDisplayID(f.second, e->x(), e->y()) == -1)
//        return;
//
//    std::shared_ptr<QMenu> context_menu(new QMenu(tr("context menu"),this));
//    std::shared_ptr<QMenu> external_menu(new QMenu(tr("External video"), this));
//    contextx = e->x();
//    contexty = e->y();
//    videoRenderer::VideoSource s = vr->getCurrentSource(f.second, contextx, contexty);
//    if(s == videoRenderer::STATIC){
//        context_menu->addAction(action_internal.get());
//        context_menu->exec(e->globalPos());
//        return;
//    }
//
//    vector<shared_ptr<QAction> > channels(external_textures.size());
//    for(int i=0; i<external_textures.size(); i++){
//        channels[i] = shared_ptr<QAction>(new QAction(QString("Channel %1").arg(QString::number(i)),this));
//        connect(channels[i].get(), &QAction::triggered, [=](){this->switch_to_external(i);});
//        external_menu->addAction(channels[i].get());
//    }
//    context_menu->addAction(action_internal.get());
//    context_menu->addMenu(external_menu.get());
//    context_menu->addAction(action_shut.get());
//    context_menu->exec(e->globalPos());
    }

}
