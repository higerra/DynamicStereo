#include "videorenderer.h"
#include "../base/quad_util.h"
#include "../base/CGHelper.h"
using namespace std;
using namespace Eigen;

namespace dynamic_stereo{
bool videoRenderer::is_shader_init = false;
GLfloat videoRenderer::highlight_stride = 0.03;
const GLfloat videoRenderer::highlight_mag = 0.2;

shared_ptr<QOpenGLShaderProgram> videoRenderer::shader =
        shared_ptr<QOpenGLShaderProgram>(new QOpenGLShaderProgram());

const int videoRenderer::video_rate = 6;

videoRenderer::videoRenderer(const std::shared_ptr<FileIO> file_io,
              const std::vector<Frame>& frames,
              const std::vector<std::vector<std::shared_ptr<QOpenGLTexture> > > &externaltextures_):
    externaltextures(externaltextures_),
    external_counter(0),
    blend_counter(0){

    initializeOpenGLFunctions();
    ifstream quadfinalfile(file_io->getQuadFinalFile().c_str());
    if(!quadfinalfile.is_open()){
        cerr << "Can not open quad config file!" <<endl;
        exit(-1);
    }
    char buffer[1024];

    quadfinalfile >> kNumTracks;
    startid.resize(kNumTracks);
    endid.resize(kNumTracks);
    videotextures.resize(kNumTracks);
    vertex_data.resize(kNumTracks);
    videoVertexBuffer.resize(kNumTracks);

    video_counter.resize(kNumTracks);
    external_counter.resize(kNumTracks);
    channel_counter.resize(kNumTracks);

    statictexture.resize(kNumTracks);
    quads.resize(kNumTracks);
    source.resize(kNumTracks);
    highlight_weight.resize(kNumTracks);
    highlight_direction.resize(kNumTracks);

    for(int tid=0; tid<kNumTracks; tid++){
        video_counter[tid] = 0;
        external_counter[tid] = 0;
        channel_counter[tid] = 0;
        source[tid] = INTERNAL;
        highlight_weight[tid] = -1;
        highlight_direction[tid] = 1;
        quadfinalfile >> startid[tid] >> endid[tid];
        vertex_data[tid].resize(endid[tid]-startid[tid]+1);
        for(int i=startid[tid]; i <= endid[tid]; i++){
            Quad curquad;
            quadfinalfile >> curquad;
            quads[tid].push_back(curquad);
            vector<double> quad_depth;
            quad_util::getQuadDepth(curquad, frames[i], quad_depth);
            for(int j=0; j<4; j++){
                const Vector2d& curpt = curquad.cornerpt[j];
                Vector2d depthpix = frames[i].RGBToDepth(curpt);
                Vector3d worldpt = frames[i].getCamera().backProject(curpt, quad_depth[j]);
                vertex_data[tid][i-startid[tid]].push_back((GLfloat)worldpt[0]);
                vertex_data[tid][i-startid[tid]].push_back((GLfloat)worldpt[1]);
                vertex_data[tid][i-startid[tid]].push_back((GLfloat)worldpt[2]);
            }
        }

        for(int texid=0; ; texid++){
            ifstream fin(file_io->getVideoTexture(tid,texid).c_str());
            if(!fin.is_open())
                break;
            fin.close();
            QImage teximg(QString::fromStdString(file_io->getVideoTexture(tid,texid)));
            for(int y=0; y<teximg.height(); ++y){
                for(int x=0; x<teximg.width(); ++x){
                    QRgb curpix = teximg.pixel(x,y);
                    QColor curcolor(curpix);
                    if(curcolor.red() <= 3 && curcolor.green() <= 3 && curcolor.blue() <= 3)
                        teximg.setPixel(x,y,qRgb(3,3,3));
                }
            }
            shared_ptr<QOpenGLTexture> curtexture(new QOpenGLTexture(teximg));
            videotextures[tid].push_back(curtexture);
        }
        //read static texture
        sprintf(buffer, "%s/video/off%03d.png", file_io->getDirectory().c_str(), tid);
        ifstream offin(buffer);
        if(!offin.is_open()){
            cerr << "Can not open static backgroud:"<<buffer<<endl;
            continue;
        }
        offin.close();
        QImage background(QString::fromStdString(string(buffer)));
        for(int y=0; y<background.height(); ++y){
            for(int x=0; x<background.width(); ++x){
                QRgb curpix = background.pixel(x,y);
                QColor curcolor(curpix);
                if(curcolor.red() <= 3 && curcolor.green() <= 3 && curcolor.blue() <= 3)
                    background.setPixel(x,y,qRgb(3,3,3));
            }
        }
        statictexture[tid] = shared_ptr<QOpenGLTexture>(new QOpenGLTexture(background));
    }
    quadfinalfile.close();

    texcoord_data.push_back(0.0); texcoord_data.push_back(0.0);
    texcoord_data.push_back(1.0); texcoord_data.push_back(0.0);
    texcoord_data.push_back(1.0); texcoord_data.push_back(1.0);
    texcoord_data.push_back(0.0); texcoord_data.push_back(1.0);


    index_data.push_back(0); index_data.push_back(1); index_data.push_back(2);
    index_data.push_back(2); index_data.push_back(3); index_data.push_back(0);


    videoVertexBuffer.resize(kNumTracks);
    for(int tid=0; tid<kNumTracks; tid++){
        videoVertexBuffer[tid].resize(endid[tid]-startid[tid]+1);
        glGenBuffers(endid[tid]-startid[tid]+1, videoVertexBuffer[tid].data());
        for(int i=0; i<videoVertexBuffer[tid].size(); i++){
            glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][i]);
            glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), vertex_data[tid][i].data(), GL_STATIC_DRAW);
        }
    }

    glGenBuffers(1, &texcoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), texcoord_data.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(GLuint), index_data.data(), GL_STATIC_DRAW);

    initializeShader();
}

void videoRenderer::initializeShader(){
    if(is_shader_init)
        return;
    if(!shader->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/video_blend.vert")){
        cerr << "videoRenderer: can not add vertex shader!" <<endl;
        exit(-1);
    }
    if(!shader->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/video_blend.frag")){
        cerr << "videoRenderer: can not add vertex shader!" <<endl;
        exit(-1);
    }
    if(!shader->link()){
        cerr << "videoRenderer: can not link shader!" <<endl;
        exit(-1);
    }
    if(!shader->bind()){
        cerr << "videoRenderer: can not bind shader!" <<endl;
        exit(-1);
    }
    shader->enableAttributeArray("vPosition");
    shader->enableAttributeArray("texcoord");
    shader->release();
    is_shader_init = true;
}

void videoRenderer::changeSource(int frameid, int x, int y, const VideoSource &new_source, int channel){
    if(new_source == EXTERNAL && externaltextures.size() == 0){
        printf("No external texture loaded\n");
        return;
    }
    int tid = getDisplayID(frameid, x, y);
    if(tid == -1)
        return;
    if(new_source == source[tid])
        return;
    printf("Changing source of track%d to %d, channel:%d\n", tid, new_source, channel);
    source[tid] = new_source;
    if(new_source == EXTERNAL){
        if(channel >= externaltextures.size()){
            cerr << "videoRenderer::changeSource: channel out of range" <<endl;
            source[tid] = INTERNAL;
            return;
        }
        channel_counter[tid] = channel;
        external_counter[tid] = 0;
    }
}

void videoRenderer::setHighlight(int frameid, int x, int y){
    int tid = getDisplayID(frameid, x, y);
    if(tid < 0){
        for(int i=0; i<kNumTracks; i++){
            highlight_weight[i] = -1.0;
        }
        return;
    }
    if(highlight_weight[tid] < -0.5){
        highlight_weight[tid] = 0.0;
        highlight_direction[tid] = 1.0;
    }
    for(int i=0; i<kNumTracks; i++){
        if(i == tid)
            continue;
        highlight_weight[i] = -1.0;
    }

}

int videoRenderer::getDisplayID(int frameid, int x, int y){
    for(int tid=0; tid<quads.size(); ++tid){
        for(int qid=0; qid<quads[tid].size(); ++qid){
            if(quads[tid][qid].frameid == frameid){
                Vector2d p((double)x, (double)y);
                if(CGHelper::isInsidePolygon(p, quads[tid][qid].cornerpt)){
                    return tid;
                }
            }
        }
    }
    return -1;
}

void videoRenderer::render(const int frameid,
                           const Navigation& navigation){
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_TEXTURE_2D);
    shader->bind();
    const QMatrix4x4& modelview = navigation.getRenderCamera();
    const QMatrix4x4& projection = navigation.getProjectionMatrix();
    shader->setUniformValue("mv_mat", modelview);
    shader->setUniformValue("mp_mat", projection);


    for(int tid=0; tid < kNumTracks; tid++){
        if(frameid < startid[tid] || frameid > endid[tid])
            continue;
        if(highlight_weight[tid] >= -0.5){
            if(highlight_weight[tid] >= highlight_mag || highlight_weight[tid] <= -0.01)
                highlight_direction[tid] *= -1;
            highlight_weight[tid] += highlight_stride * highlight_direction[tid];
            shader->setUniformValue("highlight", highlight_weight[tid]);
        }else
            shader->setUniformValue("highlight", (GLfloat)0.0);


        switch(source[tid]){
        case INTERNAL:
            renderInternal(frameid, tid,navigation);
            break;
        case EXTERNAL:
            renderExternal(frameid, tid);
            break;
        case STATIC:
            renderStatic(frameid, tid);
            break;
        default:
            renderInternal(frameid, tid,navigation);
            break;
        }
    }
    shader->release();
    glDisable(GL_TEXTURE_2D);
    glPopAttrib();
}

void videoRenderer::renderInternal(const int frameid, const int tid,const Navigation& navigation){
    const float dynamic_weight = navigation.getDynamicWeight();
    glActiveTexture(GL_TEXTURE0);
    int cur_frame = video_counter[tid];
    int next_frame = (video_counter[tid]+1) % videotextures[tid].size();
    videotextures[tid][cur_frame]->bind();
    if(!videotextures[tid][cur_frame]->isBound()){
        cerr << "videoRenderer::render: can not bind texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
        return;
    }
    glActiveTexture(GL_TEXTURE1);
    videotextures[tid][next_frame]->bind();
    if(!videotextures[tid][next_frame]->isBound()){
        cerr << "videoRenderer::render: can not bind texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
        return;
    }
    //printf("frameid %d, track %d, blend_counter:%d cur_frame:%d next_frame:%d weight:%.4f\n",frameid, tid, blend_counter, cur_frame, next_frame,
    //       navigation.getDynamicWeight());
    if(dynamic_weight <= 0.01){
        //   printf("video_counter[tid]=next_frame:%d\n", next_frame);
        video_counter[tid] = next_frame;
    }
    shader->setUniformValue("tex0",0);
    shader->setUniformValue("tex1",1);
    shader->setUniformValue("weight", dynamic_weight);
    //shader->setUniformValue("weight", (GLfloat)0.5);

    glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][frameid-startid[tid]]);
    shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);
}

void videoRenderer::renderStatic(const int frameid, const int tid){
    glActiveTexture(GL_TEXTURE0);
    statictexture[tid]->bind();
    if(!statictexture[tid]->isBound()){
        cerr << "videoRenderer::render: can not bind static texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
        return;
    }
    shader->setUniformValue("tex0",0);
    shader->setUniformValue("tex1",0);
    shader->setUniformValue("weight", (GLfloat)1.0);

    glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][frameid-startid[tid]]);
    shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);
}

void videoRenderer::renderExternal(const int frameid,const int tid){
    int channel = channel_counter[tid];
    int totalnum = (int)externaltextures[channel].size();
    if(totalnum == 0)
       return;
    external_counter[tid]++;
    if(external_counter[tid] >= totalnum)
        external_counter[tid] = 0;
    glActiveTexture(GL_TEXTURE0);
    externaltextures[channel][external_counter[tid]]->bind();
    if(!externaltextures[channel][external_counter[tid]]->isBound()){
        cerr << "videoRenderer::render: can not bind external texture: tid: "<<tid<<" frameid:"<<frameid<<endl;
        return;
    }
    shader->setUniformValue("tex0",0);
    shader->setUniformValue("tex1",0);
    shader->setUniformValue("weight", (GLfloat)1.0);

    glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer[tid][frameid-startid[tid]]);
    shader->setAttributeBuffer("vPosition", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordBuffer);
    shader->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glDrawElements(GL_TRIANGLES, (GLsizei)index_data.size(), GL_UNSIGNED_INT, 0);
}
}//namespace dynamic_rendering

