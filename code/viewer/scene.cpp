#include "scene.h"

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{
Scene::~Scene(){
    freeResource();
}

const double Scene::downsample_rate = 2.0;
shared_ptr<QOpenGLShaderProgram> Scene::shader =
        shared_ptr<QOpenGLShaderProgram>(new QOpenGLShaderProgram());
bool Scene::is_shader_init = false;

bool Scene::initialize(const std::string &path, const int scene_id_, const Navigation &navigation){
    initializeOpenGLFunctions();
    scene_id = scene_id_;
    char buffer[1024];
    sprintf(buffer, "%s/scene%03d", path.c_str(), scene_id);
    file_io = make_shared<FileIO>(FileIO(string(buffer)));
    framenum = file_io->getTotalNum();

    if(file_io->getTotalNum() == 0){
        cerr << "Empty scene:"<<buffer<<endl;
        return false;
    }

    frames.resize(framenum);
    for(int i=0; i<framenum; i++){
        frames[i].initialize(*file_io, i);
        Camera cam = navigation.getCamera(scene_id, i);
        frames[i].getCamera_nonConst().setExtrinsic(cam.getExtrinsic());
        frames[i].getCamera_nonConst().setIntrinsic(cam.getIntrinsic());
        frames[i].getDepth_nonConst().readDepthFromFile(file_io->getInpaintedDepthFile(i));
    }

    videorenderer = shared_ptr<videoRenderer>(new videoRenderer(file_io, frames, external_textures));
    allocateResource();

    glEnable(GL_TEXTURE_2D);
    textures.resize(framenum);
    for(int i=0; i<framenum; i++){
        QImage curimg(QString::fromStdString(file_io->getImage(i)));
        for(int y=0; y<curimg.height(); ++y){
            for(int x=0; x<curimg.width(); ++x){
                QRgb curpix = curimg.pixel(x,y);
                QColor curcolor(curpix);
                if(curcolor.red() <= 3 && curcolor.green() <= 3 && curcolor.blue() <= 3)
                    curimg.setPixel(x,y,qRgb(3,3,3));
            }
        }
        textures[i] = shared_ptr<QOpenGLTexture>(new QOpenGLTexture(curimg));
        //textures[i] = shared_ptr<QOpenGLTexture>(new QOpenGLTexture(
        //                   QImage(frames[i].getImage().data, frames[i].getWidth(), frames[i].getHeight(), QImage::Format_RGB888)));
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    initializeShader();
    return true;
}

void Scene::initializeShader(){
    if(is_shader_init)
        return;
    if(!shader->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/frame.vert")){
        cerr << "Scene: can not add vertex shader!" <<endl;
        exit(-1);
    }
    if(!shader->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/frame.frag")){
        cerr << "Scene: can not add vertex shader!" <<endl;
        exit(-1);
    }
    if(!shader->link()){
        cerr << "Scene: can not link shader!" <<endl;
        exit(-1);
    }
    if(!shader->bind()){
        cerr << "Scene: can not bind shader!" <<endl;
        exit(-1);
    }
    shader->enableAttributeArray("vPosition");
    shader->enableAttributeArray("texcoord");
    shader->release();
    is_shader_init = true;
}

void Scene::allocateResource(){
    const int depthwidth = frames[0].getDepth().getWidth();
    const int depthheight = frames[0].getDepth().getHeight();
    const int renderingwidth = depthwidth / downsample_rate;
    const int renderingheight = depthheight / downsample_rate;

    vertex_data.resize(framenum);
    for(int i=0; i<framenum; i++){
        for(int y=0; y<renderingheight; y++){
            for(int x=0; x<renderingwidth; x++){
                double d = frames[i].getDepth().getDepthAtInt(x * downsample_rate,y * downsample_rate);
                Vector2d RGBpix = frames[0].DepthToRGB(Vector2d(x * downsample_rate,y * downsample_rate));
                Vector3d worldpt = frames[i].getCamera().backProject(RGBpix, d);
                vertex_data[i].push_back((GLfloat)worldpt[0]);
                vertex_data[i].push_back((GLfloat)worldpt[1]);
                vertex_data[i].push_back((GLfloat)worldpt[2]);

                texcoord_data.push_back(static_cast<GLfloat>(RGBpix[0])/static_cast<GLfloat>(frames[i].getWidth()));
                texcoord_data.push_back(static_cast<GLfloat>(RGBpix[1])/static_cast<GLfloat>(frames[i].getHeight()));
            }
        }
    }

    for(int y=0; y<renderingheight-1; y++){
        for(int x=0; x<renderingwidth-1; x++){
            index_data.push_back(y*renderingwidth + x);
            index_data.push_back(y*renderingwidth + x + 1);
            index_data.push_back((y+1)*renderingwidth + x);
            index_data.push_back(y*renderingwidth + x + 1);
            index_data.push_back((y+1)*renderingwidth + x + 1);
            index_data.push_back((y+1)*renderingwidth + x);
        }
    }

    //create vertex data buffer
    vertexBuffer.resize(framenum);
    glGenBuffers(framenum, vertexBuffer.data());
    for(size_t i=0; i<vertexBuffer.size(); i++){
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer[i]);
        glBufferData(GL_ARRAY_BUFFER, vertex_data[i].size() * sizeof(GLfloat), vertex_data[i].data(), GL_STATIC_DRAW);
    }
    //create index buffer
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.size() * sizeof(GLuint), index_data.data(), GL_STATIC_DRAW);

    //texture coordinate buffer
    glGenBuffers(1, &texcoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, texcoord_data.size() * sizeof(GLfloat), texcoord_data.data(), GL_STATIC_DRAW);
}

void Scene::freeResource(){
    glDeleteBuffers(framenum, vertexBuffer.data());
    glDeleteBuffers(1, &indexBuffer);
    glDeleteBuffers(1, & texcoordBuffer);
}

void Scene::render(const int frameid, const Navigation &navigation){
    if(frameid >= framenum){
        cerr << "Scene::render(): frameid >= framenum"<<endl;
        return;
    }
    if(!shader->bind()){
        cerr << "Scene::render(): can not bind shader"<< endl;
        return;
    }
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    const QMatrix4x4& modelview = navigation.getRenderCamera();
    const QMatrix4x4& projection = navigation.getProjectionMatrix();
    shader->setUniformValue("mv_mat", modelview);
    shader->setUniformValue("mp_mat", projection);
    shader->setUniformValue("weight", (GLfloat)1.0);

    glEnable(GL_TEXTURE_2D);
    textures[frameid]->bind();
    if(!textures[frameid]->isBound())
        cerr << "Scene::render(): can not bind texture "<< endl;
    shader->setUniformValue("tex_sampler", 0);
    shader->setUniformValue("weight", (GLfloat)1.0);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer[frameid]);
    shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);

    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);

    glDisable(GL_TEXTURE_2D);
    shader->release();

    videorenderer->render(frameid, navigation);
    glFlush();
    glPopAttrib();
}

} //namespace dynamic_rendering
