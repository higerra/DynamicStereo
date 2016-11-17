#ifndef VIDEORENDERER_H
#define VIDEORENDERER_H

#include "../base/depth.h"

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QImage>
#include <vector>
#include <fstream>
#include <memory>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "navigation.h"

namespace dynamic_stereo{
    class VideoRenderer: protected QOpenGLFunctions {
    public:
        enum VideoSource{INTERNAL, EXTERNAL, STATIC};
        VideoRenderer(const std::string& name, const QImage& background, const Depth& ref_depth, const theia::Camera& camera,
                      const std::vector<Eigen::Vector2i>& loc, const cv::Mat& pixels, const std::vector<int>& corners,
                      const std::vector<std::shared_ptr<QOpenGLTexture> >* external_texture);

        void render(const Navigation &navigation);
//    void changeSource(int frameid, int x, int y,
//                      const VideoSource& new_source,
//                      int channel = 0);
//    int getDisplayID(int frameid, int x, int y);
//    void setHighlight(int frameid, int x, int y);

//    inline VideoSource getCurrentSource(int frameid, int x, int y){
//        int tid = getDisplayID(frameid, x, y);
//        if(tid == -1)
//            return STATIC;
//        return source[tid];
//    }

        inline VideoSource GetCurrentSource() const{
            return source_;
        }

    private:
        static void initializeShader();

        inline void increment_counter(){
            render_counter_ += render_direction_ ? 1 : -1;
            if(identifier_[0] == 'f'){
                if(render_counter_ == video_textures_.size()){
                    render_counter_ = 0;
                }
            }else {
                if (render_counter_ == 0 || render_counter_ == video_textures_.size() - 1) {
                    render_direction_ = !render_direction_;
                }
            }
        }

        void renderInternal(const Navigation &navigation);
//    void renderExternal(const int frameid, const int tid);
//    void renderStatic(const int frameid, const int tid);

        const std::string identifier_;

        VideoSource source_;

        std::vector<std::shared_ptr<QOpenGLTexture> > video_textures_;

        const std::vector<int> corners_;
        const std::vector<std::shared_ptr<QOpenGLTexture> >* external_textures_;
        //std::vector<std::shared_ptr<QOpenGLTexture> > static_texture_;

        std::vector<GLfloat> vertex_data_;
        GLuint video_vertex_buffer_;

        std::vector<GLfloat> texcoord_data_;
        GLuint texcoord_buffer_;

        std::vector<GLuint> index_data_;
        GLuint index_buffer_;

        static std::shared_ptr<QOpenGLShaderProgram> shader_;
        static bool is_shader_init_;

        int render_counter_;
        bool render_direction_;
        int external_counter_;
        int blend_counter_;

//    std::vector<GLfloat> highlight_weight_;
//    static GLfloat highlight_stride_;
//    static const GLfloat highlight_mag_;
//    std::vector<GLfloat> highlight_direction_;
    };

}//namespace dynamic_stereo
#endif // VIDEORENDERER_H
