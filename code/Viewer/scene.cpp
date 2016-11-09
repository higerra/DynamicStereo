#include "scene.h"

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{
    Scene::~Scene(){
        freeResource();
    }

    const double Scene::render_downsample_ = 2.0;
    shared_ptr<QOpenGLShaderProgram> Scene::shader =
            shared_ptr<QOpenGLShaderProgram>(new QOpenGLShaderProgram());
    bool Scene::is_shader_init = false;

    bool Scene::initialize(const std::string &path, const int frame_id, const Navigation &navigation){
        initializeOpenGLFunctions();
        frame_id_ = frame_id;
        file_io = make_shared<FileIO>(FileIO(path));

        CHECK_LT(frame_id_, file_io->getTotalNum());
        CHECK(background_.load(QString::fromStdString(file_io->getImage(frame_id_)))) << file_io->getImage(frame_id_);
        CHECK(depth_.readDepthFromFile(file_io->getDepthFile(frame_id_))) << file_io->getDepthFile(frame_id_);
        depth_downsample_ = (double)background_.width() / (double)depth_.getWidth();
        camera_ = navigation.GetCameraFromGlobalIndex(frame_id_);

        allocateResource();

        //read background image
        glEnable(GL_TEXTURE_2D);
         for(int y=0; y<background_.height(); ++y){
            for(int x=0; x<background_.width(); ++x){
                QRgb curpix = background_.pixel(x,y);
                QColor curcolor(curpix);
                if(curcolor.red() <= 3 && curcolor.green() <= 3 && curcolor.blue() <= 3) {
                    background_.setPixel(x, y, qRgb(3, 3, 3));
                }
            }
        }

        background_texture_ = shared_ptr<QOpenGLTexture>(new QOpenGLTexture(background_));
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

    void Scene::allocateResource() {
        const int depthwidth = depth_.getWidth();
        const int depthheight = depth_.getHeight();
        const int renderingwidth = depthwidth / render_downsample_;
        const int renderingheight = depthheight / render_downsample_;

        for (int y = 0; y < renderingheight; y++) {
            for (int x = 0; x < renderingwidth; x++) {
                double d = depth_.getDepthAtInt(x * render_downsample_, y * render_downsample_);
                Vector2d pix_loc(x * depth_downsample_ * render_downsample_,
                                 y * depth_downsample_ * render_downsample_);
                QRgb pix_color = background_.pixel(pix_loc[0], pix_loc[1]);
                Vector3d worldpt = camera_.PixelToUnitDepthRay(pix_loc) * d + camera_.GetPosition();
                vertex_data_.push_back((GLfloat) worldpt[0]);
                vertex_data_.push_back((GLfloat) worldpt[1]);
                vertex_data_.push_back((GLfloat) worldpt[2]);

                texcoord_data_.push_back(static_cast<GLfloat>(pix_loc[0]) / static_cast<GLfloat>(background_.width()));
                texcoord_data_.push_back(static_cast<GLfloat>(pix_loc[1]) / static_cast<GLfloat>(background_.height()));
            }
        }

        for (int y = 0; y < renderingheight - 1; y++) {
            for (int x = 0; x < renderingwidth - 1; x++) {
                index_data_.push_back(y * renderingwidth + x);
                index_data_.push_back(y * renderingwidth + x + 1);
                index_data_.push_back((y + 1) * renderingwidth + x);
                index_data_.push_back(y * renderingwidth + x + 1);
                index_data_.push_back((y + 1) * renderingwidth + x + 1);
                index_data_.push_back((y + 1) * renderingwidth + x);
            }
        }

        //create vertex data buffer
        glGenBuffers(1, &vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat), vertex_data_.data(), GL_STATIC_DRAW);
        //create index buffer
        glGenBuffers(1, &index_buffer_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint), index_data_.data(), GL_STATIC_DRAW);

        //texture coordinate buffer
        glGenBuffers(1, &texcoord_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
        glBufferData(GL_ARRAY_BUFFER, texcoord_data_.size() * sizeof(GLfloat), texcoord_data_.data(), GL_STATIC_DRAW);
    }

    void Scene::freeResource(){
        glDeleteBuffers(1, &vertex_buffer_);
        glDeleteBuffers(1, &index_buffer_);
        glDeleteBuffers(1, & texcoord_buffer_);
    }

    void Scene::render(const Navigation &navigation){
        CHECK(shader->bind()) << "Scene::render(): can not bind shader";

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        const QMatrix4x4& modelview = navigation.getRenderCamera();
        const QMatrix4x4& projection = navigation.getProjectionMatrix();

        shader->setUniformValue("mv_mat", modelview);
        shader->setUniformValue("mp_mat", projection);
        shader->setUniformValue("weight", (GLfloat)1.0);

        glEnable(GL_TEXTURE_2D);
        background_texture_->bind();
        CHECK(background_texture_->isBound()) << "Scene::render(): can not bind texture ";

        shader->setUniformValue("tex_sampler", 0);
        shader->setUniformValue("weight", (GLfloat)1.0);

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);

        glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
        shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glDrawElements(GL_TRIANGLES, (GLsizei)index_data_.size(), GL_UNSIGNED_INT, 0);

        glDisable(GL_TEXTURE_2D);
        shader->release();

        glFlush();
        glPopAttrib();
    }

} //namespace dynamic_rendering
