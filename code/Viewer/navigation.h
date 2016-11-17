#ifndef NAVIGATION_H
#define NAVIGATION_H
#include "animation.h"
#include "../GeometryModule/stereomodel.h"

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
#include <theia/theia.h>
#include <glog/logging.h>

namespace dynamic_stereo{

class Navigation
{
public:

    enum RenderMode{STATIC, TRANSITION_FRAME};
    enum MouseEventType{MOVE, CLICK, DOUBLE_CLICK};
    enum Direction{MOVE_FORWARD,MOVE_BACKWARD,MOVE_LEFT,MOVE_RIGHT};

    explicit Navigation(const std::string& path);
    ~Navigation();

    //read configuration file and return frame ids
    void ReadConfiguration(const std::string& json_path);

    inline const QMatrix4x4& getRenderCamera()const {return render_camera_;}
    inline const QMatrix4x4& getProjection()const {return project_camera_;}


    const theia::Camera& GetCameraFromGlobalIndex(const int idx) const;

    inline const std::vector<int>& GetFrameIdx() const{
        return frame_ids_;
    }

    inline const int getNumFrames() const{return kNumFrames;}

    inline const double GetCX() const{
        return cx;
    }
    inline const double GetCY() const{
        return cy;
    }

    inline const int GetFrameWidth() const{
        return frame_width_;
    }

    inline const int GetFrameHeight() const{
        return frame_height_;
    }

    const theia::Camera& getCamera(const int frameid)const{
        CHECK_LT(frameid, cameras_.size());
        return cameras_[frameid];
    }

    static Eigen::Vector3d interpolateVector3D(const Eigen::Vector3d& v1,
                                               const Eigen::Vector3d& v2,
                                               double percent);

    inline float getFrameWeight() const{return blendweight_frame;}
    inline int getBlendNum() const {return animation_blendNum;}

    inline RenderMode getStatus() const{return cameraStatus;}
    inline void setStatus(RenderMode mode){cameraStatus = mode;}

    inline int getCurrentFrame()const{return current_frame_;}
    inline int getNextFrame()const{return next_frame_;}


    void processKeyEvent(QKeyEvent* e);
    void processMouseEvent(MouseEventType type, int dx, int dy);
    void updateNavigation();
private:
    QMatrix4x4 getModelViewMatrix(const int frameid1,
                                   const int frameid2,
                                   const double percent) const;

    QMatrix4x4 getProjectionMatrix(const int frameid1,
                                   const int frameid2,
                                   const double percent) const;

    int getNextScene(const int base_frame, Direction direction) const;
    bool MoveFrame(Direction direction);

    int kNumFrames;
    QMatrix4x4 render_camera_;
    QMatrix4x4 project_camera_;

    double fov_;
    double cx;
    double cy;
    int frame_width_;
    int frame_height_;
    double near_plane_;
    double far_plane_;

    //for animation
    static const int animation_blendNum;
    int animation_counter;
    int video_counter;

    //for ken burns
    static const double kbspeed;
    double kbprogress;
    double kbstride;
    double kbdirection;

    float blendweight_frame;

    std::vector<int> frame_ids_;
    std::vector<theia::Camera> cameras_;
    std::vector<Eigen::Matrix<double,3,4> > extrinsics_;
    std::vector<Eigen::Vector3d> vdirs_;
    std::vector<Eigen::Vector3d> updirs_;
    std::vector<Eigen::Vector3d> camcenters_;
    std::vector<Eigen::Vector4i> paths_;

    std::vector<std::vector<float> > look_at_table_;
    RenderMode cameraStatus;

    int current_frame_;
    int next_frame_;
};
}//namespace dynamic_stereo

#endif // NAVIGATION_H
