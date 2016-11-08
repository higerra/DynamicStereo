#ifndef VIDEORENDERER_H
#define VIDEORENDERER_H
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QImage>
#include <QColor>
#include <QString>
#include <QMatrix4x4>
#include <vector>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "navigation.h"
#include "../base/file_io.h"

namespace dynamic_stereo{
class videoRenderer: protected QOpenGLFunctions {
public:
    enum VideoSource{INTERNAL, EXTERNAL, STATIC};
    videoRenderer(const std::shared_ptr<FileIO> file_io,
                  const std::vector<Frame>& frames,
                  const std::vector< std::vector<std::shared_ptr<QOpenGLTexture> > >& externaltextures_);
    void render(const int frameid,
                const Navigation &navigation);
    void changeSource(int frameid, int x, int y,
                      const VideoSource& new_source,
                      int channel = 0);
    int getDisplayID(int frameid, int x, int y);
    void setHighlight(int frameid, int x, int y);
    inline VideoSource getCurrentSource(int frameid, int x, int y){
        int tid = getDisplayID(frameid, x, y);
        if(tid == -1)
            return STATIC;
        return source[tid];
    }

private:
    static void initializeShader();
    void renderInternal(const int frameid, const int tid, const Navigation &navigation);
    void renderExternal(const int frameid, const int tid);
    void renderStatic(const int frameid, const int tid);

    std::vector<VideoSource> source;
    int kNumTracks;
    std::vector<std::vector<Quad> > quads;
    std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > > videotextures;
    const std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > >& externaltextures;
    std::vector<std::shared_ptr<QOpenGLTexture> > statictexture;

    std::vector<std::vector<std::vector<GLfloat> > > vertex_data;
    std::vector<std::vector<GLuint> > videoVertexBuffer;

    std::vector<GLfloat> texcoord_data;
    GLuint texcoordBuffer;

    std::vector<GLuint> index_data;
    GLuint indexBuffer;

    static std::shared_ptr<QOpenGLShaderProgram> shader;
    static bool is_shader_init;
    std::vector<int> startid, endid;
    std::vector<int> video_counter;
    std::vector<int> external_counter;
    std::vector<int> channel_counter;

    int blend_counter;
    static const int video_rate;

    std::vector<GLfloat> highlight_weight;
    static GLfloat highlight_stride;
    static const GLfloat highlight_mag;
    std::vector<GLfloat> highlight_direction;
};

}//namespace dynamic_stereo
#endif // VIDEORENDERER_H
