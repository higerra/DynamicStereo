#include <QRect>
#include <QPoint>

#include "videorenderer.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{
    bool VideoRenderer::is_shader_init_ = false;

//    GLfloat videoRenderer::highlight_stride = 0.03;
//    const GLfloat videoRenderer::highlight_mag = 0.2;

    shared_ptr<QOpenGLShaderProgram> VideoRenderer::shader_ =
            shared_ptr<QOpenGLShaderProgram>(new QOpenGLShaderProgram());


    VideoRenderer::VideoRenderer(const std::string& name, const QImage& background, const Depth& ref_depth, const theia::Camera& camera,
                                 const std::vector<Eigen::Vector2d>& loc, const cv::Mat& pixels,
                                 const std::vector<std::shared_ptr<QOpenGLTexture> >* external_texture):
            identifier_(name), source_(INTERNAL), external_textures_(external_texture), render_counter_(0), external_counter_(0), blend_counter_(0) {
        //compute the bounding box
        //x_min, x_max, y_min, y_max
        LOG(INFO) << "Initializing dynamic region: " << name;
        vector<int> roi{-1, -1, -1, -1};
        for (const auto &pt: loc) {
            if (roi[0] < 0 || pt[0] < roi[0]) {
                roi[0] = pt[0];
            }
            if (roi[1] < 0 || pt[0] > roi[1]) {
                roi[1] = pt[0];
            }
            if (roi[2] < 0 || pt[1] < roi[2]) {
                roi[2] = pt[1];
            }
            if (roi[3] < 0 || pt[1] > roi[3]) {
                roi[2] = pt[1];
            }
        }
        QRect roi_rect(QPoint(roi[0], roi[2]), QPoint(roi[1], roi[3]));
        const int kFrame = pixels.rows;

        //fill texture data
        video_textures_.resize(kFrame);
        for (auto i = 0; i < video_textures_.size(); ++i) {
            QImage sub_img = background.copy(roi_rect);
            for (auto pid = 0; pid < loc.size(); ++pid) {
                const int x = loc[pid][0] - roi_rect.left();
                const int y = loc[pid][1] - roi_rect.top();
                cv::Vec3b pv = pixels.at<Vec3b>(i, pid);
                sub_img.setPixel(x, y, qRgb((int) pv[0], (int) pv[1], (int) pv[2]));
                video_textures_[i].reset(new QOpenGLTexture(sub_img));
            }
        }

        //fill vertex data
        for (int y = roi_rect.top(); y < roi_rect.bottom(); ++y) {
            for (int x = roi_rect.left(); x < roi_rect.right(); ++x) {
                int x2 = x - roi_rect.left();
                int y2 = y - roi_rect.top();
                double d = ref_depth.getDepthAtInt(x, y);
                Vector3d world_pt = camera.PixelToUnitDepthRay(Vector2d(x, y)) * d + camera.GetPosition();
                vertex_data_.push_back((GLfloat) world_pt[0]);
                vertex_data_.push_back((GLfloat) world_pt[1]);
                vertex_data_.push_back((GLfloat) world_pt[2]);

                texcoord_data_.push_back((float) x2 / (float) roi_rect.width());
                texcoord_data_.push_back((float) y2 / (float) roi_rect.height());
            }
        }

        //fill vertex index data
        for (const auto &pt: loc) {
            int x = pt[0] - roi_rect.left();
            int y = pt[1] - roi_rect.top();
            if (x == roi_rect.right() || y == roi_rect.bottom()) {
                continue;
            }
            index_data_.push_back(y * roi_rect.width() + x);
            index_data_.push_back(y * roi_rect.width() + x + 1);
            index_data_.push_back((y + 1) * roi_rect.width() + x);
            index_data_.push_back(y * roi_rect.width() + x + 1);
            index_data_.push_back((y + 1) * roi_rect.width() + x + 1);
            index_data_.push_back((y + 1) * roi_rect.width() + x);
        }

        //create OpenGL buffers
        glGenBuffers(1, &video_vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, video_vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat), vertex_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &index_buffer_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint), index_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &texcoord_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
        glBufferData(GL_ARRAY_BUFFER, texcoord_data_.size() * sizeof(GLfloat), texcoord_data_.data(), GL_STATIC_DRAW);

        initializeShader();
    }

    void VideoRenderer::initializeShader() {
        if (is_shader_init_)
            return;
        CHECK(!shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/frame.vert"))
        << "videoRenderer: can not add vertex shader!";
        CHECK(!shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/frame.frag"))
        << "can not add vertex shader!";
        CHECK(!shader_->link()) << "videoRenderer: can not link shader!";
        CHECK(!shader_->bind()) << "videoRenderer: can not bind shader!";
        shader_->enableAttributeArray("vPosition");
        shader_->enableAttributeArray("texcoord");
        shader_->release();
        is_shader_init_ = true;
    }

    void VideoRenderer::render(const int frameid,
                               const Navigation& navigation){
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_TEXTURE_2D);
        shader_->bind();
        const QMatrix4x4& modelview = navigation.getRenderCamera();
        const QMatrix4x4& projection = navigation.getProjection();
        shader_->setUniformValue("mv_mat", modelview);
        shader_->setUniformValue("mp_mat", projection);
        shader_->setUniformValue("weight", (GLfloat) 1.0);

        switch(source_){
            case INTERNAL:
                renderInternal(navigation);
                break;
            default:
                renderInternal(navigation);
                break;
        }
        shader_->release();
        glDisable(GL_TEXTURE_2D);
        glPopAttrib();
    }

    void VideoRenderer::renderInternal(const Navigation& navigation){
        glActiveTexture(GL_TEXTURE0);
        video_textures_[render_counter_]->bind();
        if(!video_textures_[render_counter_]->isBound()){
            cerr << "videoRenderer::render: can not bind texture " << render_counter_;
            return;
        }
        shader_->setUniformValue("tex0",0);

        glBindBuffer(GL_ARRAY_BUFFER, video_vertex_buffer_);
        shader_->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
        glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
        shader_->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glDrawElements(GL_TRIANGLES, (GLsizei)index_data_.size(), GL_UNSIGNED_INT, 0);
    }

//    void videoRenderer::renderStatic(const int frameid, const int tid){
//        glActiveTexture(GL_TEXTURE0);
//        statictexture[tid]->bind();
//        if(!statictexture[tid]->isBound()){
//            cerr << "videoRenderer::render: can not bind static texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
//            return;
//        }
//        shader->setUniformValue("tex0",0);
//        shader->setUniformValue("tex1",0);
//        shader->setUniformValue("weight", (GLfloat)1.0);
//
//        glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][frameid-startid[tid]]);
//        shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
//        glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
//        shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
//        glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);
//    }
//
//    void videoRenderer::renderExternal(const int frameid,const int tid){
//        int channel = channel_counter[tid];
//        int totalnum = (int)externaltextures[channel].size();
//        if(totalnum == 0)
//            return;
//        external_counter[tid]++;
//        if(external_counter[tid] >= totalnum)
//            external_counter[tid] = 0;
//        glActiveTexture(GL_TEXTURE0);
//        externaltextures[channel][external_counter[tid]]->bind();
//        if(!externaltextures[channel][external_counter[tid]]->isBound()){
//            cerr << "videoRenderer::render: can not bind external texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
//            return;
//        }
//        shader->setUniformValue("tex0",0);
//        shader->setUniformValue("tex1",0);
//        shader->setUniformValue("weight", (GLfloat)1.0);
//
//        glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][frameid-startid[tid]]);
//        shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
//        glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
//        shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
//        glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);
//    }
}//namespace dynamic_rendering

