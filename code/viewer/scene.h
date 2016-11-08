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
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <memory>
#include "navigation.h"
#include "animation.h"
#include "../base/file_io.h"

namespace dynamic_stereo{

class Scene : protected QOpenGLFunctions{
public:
    Scene(const std::vector< std::vector<std::shared_ptr<QOpenGLTexture> > >& external_textures_)
        :external_textures(external_textures_), framenum(0), scene_id(0){}
    ~Scene();
    bool initialize(const std::string& path, const int scene_id, const Navigation& navigation);

    void render(const int frameid, const Navigation& navigation);
    inline const std::shared_ptr<videoRenderer>& getVideoRenderer() const{
        return videorenderer;
    }

private:
    static void initializeShader();

    const std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > >& external_textures;
    std::shared_ptr<FileIO> file_io;
    int framenum;
    int scene_id;
    static const double downsample_rate;

    std::vector<cv::Mat> frames;
    std::vector<std::vector<GLfloat> > vertex_data;
    std::vector<GLuint> index_data;
    std::vector<GLfloat> texcoord_data;

    GLuint indexBuffer;
    GLuint dynamicIndexBuffer;
    std::vector<GLuint> vertexBuffer;
    GLuint texcoordBuffer;
    std::vector<std::shared_ptr<QOpenGLTexture> > textures;

    static std::shared_ptr<QOpenGLShaderProgram> shader;
    static bool is_shader_init;

    std::shared_ptr<videoRenderer> videorenderer;
    void allocateResource();
    void freeResource();
};

}//namespace dynamic_stereo


#endif // SCENE_H
