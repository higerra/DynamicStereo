#ifndef NAVIGATION_H
#define NAVIGATION_H
#include "animation.h"

#include <iostream>
#include <vector>
#include <mutex>
#include <memory>
#include <string>

#include <QKeyEvent>
#include <QMatrix4x4>
#include <QVector3D>

#include <QKeyEvent>
#include <QMouseEvent>
#include <Eigen/Eigen>


namespace dynamic_stereo{

class Navigation
{
public:

    enum RenderMode{STATIC, TRANSITION_FRAME, TRANSITION_SCENE};
    enum MouseEventType{MOVE, CLICK, DOUBLE_CLICK};

    explicit Navigation(const std::string& path);
    ~Navigation();

    inline const QMatrix4x4& getRenderCamera()const {return renderCamera;}
    inline const QMatrix4x4& getProjectionMatrix()const{
        return projectionMatrix;
    }

    inline const int getNumScenes() const{return kNumScenes;}
    const Eigen::Matrix4d& getSceneTransformation(const int id)const{
        if(id >= scene_transform.size()){
            std::cerr << "Navigation::getSceneTransformation: id out of bound!"<<endl;
            exit(-1);
        }
        return scene_transform[id];
    }

    const Camera& getCamera(const int sceneid, const int frameid)const{
        if(sceneid >= cameras.size()){
            std::cerr << "Navigation::getCamera(): scene id out of bound!" << endl;
            exit(-1);
        }
        if(frameid >= cameras[sceneid].size()){
            std::cerr << "Navigation::getCamera(): frame id out of bound!" << endl;
            exit(-1);
        }
        return cameras[sceneid][frameid];
    }

    static Eigen::Vector3d interpolateVector3D(const Eigen::Vector3d& v1,
                                               const Eigen::Vector3d& v2,
                                               double percent);

    inline float getFrameWeight() const{return blendweight_frame;}
    inline float getDynamicWeight() const{return blendweight_dynamic;}
    inline int getBlendNum() const {return animation_blendNum;}

    inline RenderMode getStatus() const{return cameraStatus;}
    inline void setStatus(RenderMode mode){cameraStatus = mode;}

    inline std::pair<int,int> getCurrentFrame()const{return current_frame;}
    inline std::pair<int,int> getNextFrame()const{return next_frame;}


    void processKeyEvent(QKeyEvent* e);
    void processMouseEvent(MouseEventType type, int dx, int dy);
    void updateNavigation();
private:
    void getLookAtParam(const size_t sceneid, const size_t frameid, std::vector<float>& param);
    QMatrix4x4 getModelViewMatrix(const size_t sceneid1, const size_t frameid1,
                                  const size_t sceneid2, const size_t frameid2,
                                  const double percent);
    QMatrix4x4 getModelViewMatrix(const std::pair<int,int>& currentframe,
                                   const std::pair<int,int>& nextframe,
                                   const double percent){
        return getModelViewMatrix(currentframe.first, currentframe.second,
                                  nextframe.first, nextframe.second,
                                  percent);
    }

    std::pair<int, int> getNextScene(const std::pair<int, int> &base_frame, int direction) const;
    void rotate(int stride, int direction);
    bool moveScene(int direction);

    void updateDynamicWeight();

    std::vector<std::vector<Camera> > cameras;
    std::vector<Eigen::Vector3d> scene_centers;
    std::vector<std::vector<int> > walkablepath;
    std::vector<Eigen::Matrix4d> scene_transform;
    int kNumScenes;
    QMatrix4x4 projectionMatrix;
    QMatrix4x4 renderCamera;

    static const int cx;
    static const int cy;
    double fov;

    //for animation
    static const int animation_stride;
    static const int animation_blendNum;
    static const int speed_move_scene;
    static const int video_rate;
    int animation_counter;
    int video_counter;

    //for ken burns
    static const double kbspeed;
    double kbprogress;
    double kbstride;
    double kbdirection;

    float blendweight_frame;
    float blendweight_dynamic;

    std::vector<std::vector<std::vector<float> > >lookAtTable;
    std::vector<std::vector<Eigen::Vector3d> > vdirs;
    std::vector<std::vector<Eigen::Vector3d> > updirs;
    std::vector<std::vector<Eigen::Vector3d> > camcenters;
    RenderMode cameraStatus;

    std::pair<int,int> current_frame;
    std::pair<int,int> next_frame;
};
}//namespace dynamic_stereo

#endif // NAVIGATION_H
