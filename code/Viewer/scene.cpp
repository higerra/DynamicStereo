#include "scene.h"
#include "../Cinemagraph/cinemagraph.h"

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{
    Scene::~Scene(){
        freeResource();
    }

    const double Scene::render_downsample_ = 1.0;
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
        //depth_.fillholeAndSmooth(0.1);

        depth_.updateStatics();


        depth_downsample_ = (double)background_.width() / (double)depth_.getWidth();
        camera_ = navigation.GetCameraFromGlobalIndex(frame_id_);

        LOG(INFO) << "Initializing static model";
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

        //read cinemagraph
        LOG(INFO) << "Initializing dynamic renderers";
        char buffer[128] = {};
        sprintf(buffer, "%s/temp/cinemagraph_%05d_RPCA.cg", file_io->getDirectory().c_str(), frame_id_);
        Cinemagraph::Cinemagraph cinemagraph;
        Cinemagraph::ReadCinemagraph(string(buffer), cinemagraph);
        for(int i=0; i<cinemagraph.pixel_loc_flashy.size(); ++i){
            sprintf(buffer, "flashy_%03d", i);
            std::shared_ptr<VideoRenderer> dynamic_renderer(
                    new VideoRenderer(string(buffer), background_, depth_, camera_,
                                      cinemagraph.pixel_loc_flashy[i], cinemagraph.pixel_mat_flashy[i], {}, nullptr));
            video_renderers_.push_back(dynamic_renderer);
        }
        for(int i=0; i<cinemagraph.pixel_loc_display.size(); ++i){
            sprintf(buffer, "display_%03d", i);
            std::shared_ptr<VideoRenderer> dynamic_renderer(
                    new VideoRenderer(string(buffer), background_, depth_, camera_,
                                      cinemagraph.pixel_loc_display[i], cinemagraph.pixel_mat_display[i],
                                      cinemagraph.corners[i], nullptr));
            video_renderers_.push_back(dynamic_renderer);
        }

        return true;
    }

    void Scene::initializeShader(){
        if(is_shader_init)
            return;
        CHECK(shader->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/frame.vert")) << "Scene: can not add vertex shader!";
        CHECK(shader->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/frame.frag")) << "Scene: can not add vertex shader!";
        CHECK(shader->link()) << "Scene: can not link shader!";
        CHECK(shader->bind()) << "Scene: can not bind shader!";
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

        const double max_depth_diff = (depth_.getMaxDepth() - depth_.getMinDepth()) + 1;

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
                double d1 = depth_.getDepthAtInt(x * render_downsample_, y * render_downsample_);
                double d2 = depth_.getDepthAtInt((x + 1) * render_downsample_, y * render_downsample_);
                double d3 = depth_.getDepthAtInt((x + 1) * render_downsample_, (y + 1) * render_downsample_);
                double d4 = depth_.getDepthAtInt(x * render_downsample_, (y + 1) * render_downsample_);
                if(std::fabs(d1 - d2) < max_depth_diff && std::fabs(d1 - d4) < max_depth_diff && std::fabs(d2 - d4) < max_depth_diff) {
                    index_data_.push_back(y * renderingwidth + x);
                    index_data_.push_back(y * renderingwidth + x + 1);
                    index_data_.push_back((y + 1) * renderingwidth + x);
                }
                if(std::fabs(d2 - d3) < max_depth_diff && std::fabs(d3 - d4) < max_depth_diff && std::fabs(d2 - d4) < max_depth_diff) {
                    index_data_.push_back(y * renderingwidth + x + 1);
                    index_data_.push_back((y + 1) * renderingwidth + x + 1);
                    index_data_.push_back((y + 1) * renderingwidth + x);
                }
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
        const QMatrix4x4& projection = navigation.getProjection();

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

        for(const auto video_renderer: video_renderers_){
            video_renderer->render(navigation);
        }
        glFlush();
        glPopAttrib();

    }

} //namespace dynamic_rendering
