#ifndef SCENE_H
#define SCENE_H
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <QGLWidget>
#include <QOpenGLContext>
#include <QTimer>
#include <QColor>
#include <QImage>
#include <QFile>
#include <QJsonDocument>

#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <memory>
#include "navigation.h"
#include "animation.h"
#include "videorenderer.h"
#include "../base/file_io.h"
#include "../base/depth.h"

namespace dynamic_stereo{

    class Scene : protected QOpenGLFunctions{
    public:
        Scene(const std::vector< std::vector<std::shared_ptr<QOpenGLTexture> > >& external_textures)
                :external_textures_(external_textures), frame_id_(0){}
        ~Scene();
        bool initialize(const std::string& path, const std::string& cinemagraph_type,
                        const int frame_id, const Navigation& navigation);

        void render(const Navigation& navigation);

    private:
        static void initializeShader();

        const std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > >& external_textures_;
        std::shared_ptr<FileIO> file_io;
        int frame_id_;
        static const double render_downsample_;
        double depth_downsample_;

        QImage background_;
        Depth depth_;
        theia::Camera camera_;

        std::vector<std::shared_ptr<VideoRenderer> > video_renderers_;

        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        std::vector<GLfloat> texcoord_data_;

        GLuint index_buffer_;
        GLuint dynamicIndexBuffer;
        GLuint vertex_buffer_;
        GLuint texcoord_buffer_;
        std::shared_ptr<QOpenGLTexture> background_texture_;

        static std::shared_ptr<QOpenGLShaderProgram> shader;
        static bool is_shader_init;

        void allocateResource();
        void freeResource();
    };

}//namespace dynamic_stereo


#endif // SCENE_H
