#include "navigation.h"
#include <iostream>
#include <fstream>
#include <string>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QImage>

#include "../base/file_io.h"
#include "../base/depth.h"

#define PI 3.1415927

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{

    const int Navigation::animation_blendNum = 12;

    Navigation::Navigation(const string& path):
            fov_(35.0),
            cx(-1),
            cy(-1),
            frame_width_(-1),
            frame_height_(-1),
            near_plane_(-1),
            far_plane_(-1),
            cameraStatus(STATIC),
            current_frame_(0),
            next_frame_(0),
            animation_counter(0),
            video_counter(0),
            kbprogress(0.0),
            kbstride(0.02),
            kbdirection(1.0),
            blendweight_frame(1.0){

        FileIO file_io(path);
        //read sfm model
        SfMModel sfm_model;
        char buffer[1024];
        sprintf(buffer, "%s/sfm/reconstruction.recon", path.c_str());
        sfm_model.init(string(buffer));

        QImage sample_img(QString::fromStdString(file_io.getImage(0)));
        CHECK(sample_img.bits());
        frame_width_ = sample_img.width();
        frame_height_ = sample_img.height();
        //read configuration file and get reference frames
        sprintf(buffer, "%s/render.json", path.c_str());
        ReadConfiguration(string(buffer));


        kNumFrames = frame_ids_.size();
        LOG(INFO) << "Number of frames: " << kNumFrames;

        string tempstring;
        Matrix3d iden3 = Matrix3d::Identity();
        //read scene transformation
        for(int i=0; i<kNumFrames; i++) {
            theia::Camera curcam = sfm_model.getCamera(frame_ids_[i]);
            cameras_.push_back(curcam);
            camcenters_.push_back(curcam.GetPosition());

//            cx = sfm_model.getCamera(i).PrincipalPointX();
//            cy = sfm_model.getCamera(i).PrincipalPointY();
            cx = frame_width_ / 2;
            cy = frame_height_ / 2;


            Vector3d vdir = curcam.PixelToUnitDepthRay(Vector2d(cx, cy));
            vdirs_.push_back(vdir);

            Vector3d updir = curcam.PixelToUnitDepthRay(Vector2d(frame_width_/2, 0)) -
                    curcam.PixelToUnitDepthRay(Vector2d(frame_width_/2, frame_height_/2));
            updir.normalize();

            updirs_.push_back(updir);
        }

        for(const auto fid: frame_ids_){
            Depth cur_depth;
            cur_depth.readDepthFromFile(file_io.getDepthFile(fid));
            cur_depth.updateStatics();
            far_plane_ = std::max(far_plane_, cur_depth.getMaxDepth());
            if(near_plane_ < 0 || cur_depth.getMinDepth() < near_plane_){
                near_plane_ = cur_depth.getMinDepth();
            }
        }

        //set initial camera
        render_camera_ = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
        project_camera_ = getProjectionMatrix(current_frame_, next_frame_, blendweight_frame);
    }

    Navigation::~Navigation(){

    }

    void Navigation::ReadConfiguration(const std::string& json_path){
        LOG(INFO) << "Reading " << json_path;
        QFile json_file(QString::fromStdString(json_path));
        CHECK(json_file.open(QIODevice::ReadOnly));

        QByteArray json_data = json_file.readAll();
        QJsonDocument json_doc = QJsonDocument::fromJson(json_data);
        QJsonObject json_obj = json_doc.object();

        fov_ = json_obj[QString("fov")].toDouble();

        QJsonArray frame_array = json_obj[QString("frames")].toArray();
        for(const auto& frame: frame_array){
            QJsonObject frame_obj = frame.toObject();
            frame_ids_.push_back(frame_obj[QString("frameid")].toInt());
        }

        paths_.resize(frame_ids_.size(), Eigen::Vector4i(-1,-1,-1,-1));
        int index = 0;
        QJsonArray path_array = json_obj[QString("paths")].toArray();
        CHECK_EQ(path_array.size(), frame_ids_.size());
        for(const auto& path: path_array) {
            QJsonArray cur_path = path.toArray();
            CHECK_EQ(cur_path.size(), 4);
            for (int i = 0; i < 4; ++i) {
                paths_[index][i] = cur_path[i].toInt();
            }
            index++;
        }
    }

    const theia::Camera& Navigation::GetCameraFromGlobalIndex(const int idx) const{
        for(int i=0; i<frame_ids_.size(); ++i){
            if(frame_ids_[i] == idx){
                return cameras_[i];
            }
        }
        CHECK(true) << "Index not found " << idx;
    }

    QMatrix4x4 Navigation::getModelViewMatrix(const int frameid1,
                                              const int frameid2,
                                              const double percent) const{
        CHECK_LT(frameid1, kNumFrames);
        CHECK_LT(frameid2, kNumFrames);

        Vector3d camcenter = percent * camcenters_[frameid1] + (1.0-percent) * camcenters_[frameid2];
        Vector3d updir = interpolateVector3D(updirs_[frameid1], updirs_[frameid2], 1.0 - percent);
        Vector3d framecenter = (vdirs_[frameid1] + camcenters_[frameid1]) * percent +
                               (1-percent) * (vdirs_[frameid2] + camcenters_[frameid2]);

        QMatrix4x4 m;
        m.setToIdentity();
        m.lookAt(QVector3D(camcenter[0], camcenter[1], camcenter[2]),
                 QVector3D(framecenter[0], framecenter[1], framecenter[2]),
                 QVector3D(updir[0], updir[1], updir[2]));

        return m;
    }

    QMatrix4x4 Navigation::getProjectionMatrix(const int frameid1, const int frameid2, const double percent) const{
        CHECK_LT(frameid1, kNumFrames);
        CHECK_LT(frameid2, kNumFrames);

        const double focal1 = cameras_[frameid1].FocalLength();
        const double focal2 = cameras_[frameid2].FocalLength();
        double fov = std::atan(1/((focal1 * percent + (1 - percent) * focal2))) /
                     std::atan(1/cameras_[0].FocalLength()) *
                     fov_;

        QMatrix4x4 cur_proj;
        cur_proj.setToIdentity();
        cur_proj.perspective(fov, (float)frame_width_ / (float)frame_height_, near_plane_/2, far_plane_ * 2);
        return cur_proj;
    }

    Vector3d Navigation::interpolateVector3D(const Vector3d &v1,
                                             const Vector3d &v2,
                                             double percent){
        const double epsilon = std::numeric_limits<double>::epsilon();
        if(percent < epsilon) return v1;
        if(percent > 1-epsilon) return v2;
        double theta = percent * std::acos(v1.dot(v2));
        Vector3d k = v1.cross(v2);
        k.normalize();
        //follow by rodrigues rotation formula
        Vector3d res = v1*std::cos(theta) + (k.cross(v1))*std::sin(theta) + k*(k.dot(v1))*(1-std::cos(theta));
        return res;
    }

    void Navigation::updateNavigation() {
        if (cameraStatus == TRANSITION_FRAME) {
            animation_counter++;
            if (animation_counter >= animation_blendNum) {
                cameraStatus = STATIC;
                animation_counter = 0;
                blendweight_frame = 1.0;
                current_frame_ = next_frame_;
                render_camera_ = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
                project_camera_ = getProjectionMatrix(current_frame_, next_frame_, blendweight_frame);
            } else {
                blendweight_frame = animation::getFramePercent(animation_counter, animation_blendNum);
                if (blendweight_frame < 0.001) {
                    printf("blendweight_frame:%.4f\n", blendweight_frame);
                }
                render_camera_ = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
                project_camera_ = getProjectionMatrix(current_frame_, next_frame_, blendweight_frame);
            }
        }
    }

    int Navigation::getNextScene(const int base_frame, Direction direction) const{
        if(getNumFrames() == 1)
            return -1;
        //search optimal next frame by ray tracing

        //toy implementation
//        if(direction == MOVE_FORWARD || direction == MOVE_RIGHT){
//            if(base_frame < kNumFrames - 1) {
//                return base_frame + 1;
//            }else{
//                return 0;
//            }
//        }
//        if(direction == MOVE_BACKWARD || direction == MOVE_LEFT){
//            if(base_frame > 0) {
//                return base_frame - 1;
//            }else{
//                return kNumFrames - 1;
//            }
//        }

        int next_frame = base_frame;
        if(direction == MOVE_FORWARD){
            next_frame = paths_[base_frame][0];
        }else if(direction == MOVE_BACKWARD){
            next_frame = paths_[base_frame][1];
        }else if(direction == MOVE_LEFT){
            next_frame = paths_[base_frame][2];
        }else if(direction == MOVE_RIGHT){
            next_frame = paths_[base_frame][3];
        }
        return next_frame;
    }

    bool Navigation::MoveFrame(Direction direction){
        int temp_next = getNextScene(current_frame_,direction);
        if(temp_next < 0) {
            return false;
        }
        current_frame_ = next_frame_;
        next_frame_ = temp_next;
        animation_counter = 0;
        return true;
    }

    void Navigation::processKeyEvent(QKeyEvent* e){
        if(cameraStatus != STATIC)
            return;
        switch(e->key()){
            case Qt::Key_Left:
                if(MoveFrame(MOVE_LEFT)) {
                    cameraStatus = TRANSITION_FRAME;
                }
                break;
            case Qt::Key_Right:
                if(MoveFrame(MOVE_RIGHT)) {
                    cameraStatus = TRANSITION_FRAME;
                }
                break;
            case Qt::Key_Up:{
                if(MoveFrame(MOVE_FORWARD)) {
                    cameraStatus = TRANSITION_FRAME;
                }
                break;
            }
            case Qt::Key_Down:{
                if(MoveFrame(MOVE_BACKWARD)) {
                    cameraStatus = TRANSITION_FRAME;
                }
                break;
            }
            default:
                break;
        }
    }

    void Navigation::processMouseEvent(MouseEventType type, int dx, int dy){
        if(cameraStatus != STATIC)
            return;
        switch(type){
            case(CLICK):
                break;
            default:
                break;
        }
    }

}//namespace dynamic_rendering

