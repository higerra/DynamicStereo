#include "navigation.h"
#include <iostream>
#include <fstream>
#include <string>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "../base/file_io.h"
#include "../base/plane3D.h"


#define PI 3.1415927

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{

    const int Navigation::animation_stride = 15;
    const int Navigation::animation_blendNum = 24;
    const int Navigation::video_rate = 6;
    const double Navigation::kbspeed = 2.0;
    const int Navigation::speed_move_scene = 2;

    Navigation::Navigation(const string& path):
            fov(35.0),
            cx(-1),
            cy(-1),
            cameraStatus(STATIC),
            current_frame_(0),
            next_frame_(0),
            animation_counter(0),
            video_counter(0),
            kbprogress(0.0),
            kbstride(0.02),
            kbdirection(1.0),
            blendweight_frame(1.0){

        //read sfm model
        SfMModel sfm_model;
        char buffer[1024];
        sprintf(buffer, "%s/sfm/reconstruction.recon", path.c_str());
        sfm_model.init(string(path));

        //read configuration file and get reference frames
        sprintf(buffer, "%s/conf.json", path.c_str());
        ReadConfiguration(string(buffer));

        kNumFrames = frame_ids_.size();
        cameras_.resize(kNumFrames);
        vdirs_.resize(kNumFrames);
        updirs_.resize(kNumFrames);
        camcenters_.resize(kNumFrames);
        extrinsics_.resize(kNumFrames);

        string tempstring;
        Matrix3d iden3 = Matrix3d::Identity();
        const double kFarDepth = 10.0;
        //read scene transformation
        for(int i=0; i<kNumFrames; i++) {
            theia::Camera curcam = sfm_model.getCamera(frame_ids_[i]);
            if(cx < 0 || cy < 0){
                cx = curcam.PrincipalPointX();
                cy = curcam.PrincipalPointY();
            }
            cameras_.push_back(curcam);
            camcenters_.push_back(curcam.GetPosition());
            Vector3d vdir = curcam.PixelToUnitDepthRay(Vector2d(cx,cy));
            vdir.normalize();
            vdirs_.push_back(vdir);

            theia::Matrix3x4d extrinsic;
            theia::ComposeProjectionMatrix(iden3, curcam.GetOrientationAsAngleAxis(), curcam.GetPosition(), &extrinsic);
            extrinsics_.push_back(extrinsic);
            Vector3d updir = extrinsic * Vector4d(0.0, -1.0, 0.0, 1.0) - curcam.GetPosition();
            updir.normalize();
            updirs_.push_back(updir);

//            //fill in the look at table
//            look_at_table_[i].resize(9);
//            for(int j=0; j<3; ++j){
//                look_at_table_[i][j] = static_cast<float>(camcenters[i][j]);
//            }
//            for(int j=3; j<6; ++j){
//                look_at_table_[i][j] = static_cast<float>(camcenters[i][j-3] + vdirs[i][j-3] * kFarDepth);
//            }
//            for(int j=6; j<9; ++j){
//                look_at_table_[i][j] = static_cast<float>(updirs[i][j-6]);
//            }
        }

        //set initial camera
        projectionMatrix.setToIdentity();
        projectionMatrix.perspective(fov, (float)cx / (float) cy, 0.0, 100);
        renderCamera = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
    }

    Navigation::~Navigation(){

    }

    void Navigation::ReadConfiguration(const std::string& json_path){
        QFile json_file(QString::fromStdString(json_path));
        CHECK(json_file.open(QIODevice::ReadOnly));

        QByteArray json_data = json_file.readAll();
        QJsonDocument json_doc = QJsonDocument::fromJson(json_data);
        QJsonObject json_obj = json_doc.object();

        QJsonArray frame_array = json_obj["frames"].toArray();
        for(const auto& frame: frame_array){
            QJsonObject frame_obj = frame.toObject();
            frame_ids_.push_back(frame_obj["frameid"].toInt());
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

    QMatrix4x4 Navigation::getModelViewMatrix(const size_t frameid1,
                                              const size_t frameid2,
                                              const double percent){
        CHECK_LT(frameid1, kNumFrames);
        CHECK_LT(frameid2, kNumFrames);

        const Vector3d& center1 = camcenters_[frameid1];
        const Vector3d& center2 = camcenters_[frameid2];

        const Vector3d& updir1 = updirs_[frameid1];
        const Vector3d& updir2 = updirs_[frameid2];
        const Vector3d& vdir1 = vdirs_[frameid1];
        const Vector3d& vdir2 = vdirs_[frameid2];

        Vector3d camcenter = percent * center1 + (1.0-percent) * center2;
        Vector3d updir = interpolateVector3D(updir1, updir2, 1 - percent);
        Vector3d vdir = interpolateVector3D(vdir1, vdir2, 1 - percent);
        Vector3d framecenter = vdir + camcenter;

        QMatrix4x4 m;
        m.setToIdentity();
        m.lookAt(QVector3D(camcenter[0], camcenter[1], camcenter[2]),
                 QVector3D(framecenter[0], framecenter[1], framecenter[2]),
                 QVector3D(updir[0], updir[1], updir[2]));
        return m;
    }

    Vector3d Navigation::interpolateVector3D(const Vector3d &v1,
                                             const Vector3d &v2,
                                             double percent){
        const double epsilon = 0.001;
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
            if (animation_counter >= animation_blendNum - 1) {
                cameraStatus = STATIC;
                animation_counter = 0;
                blendweight_frame = 1.0;
                current_frame_ = next_frame_;
                renderCamera = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
            } else {
                if (cameraStatus == TRANSITION_FRAME)
                    blendweight_frame = animation::getFramePercent(animation_counter, animation_blendNum);
                else
                    blendweight_frame = animation::getFramePercent(animation_counter, animation_blendNum);
                if (blendweight_frame < 0.001) {
                    printf("blendweight_frame:%.4f\n", blendweight_frame);
                }
                renderCamera = getModelViewMatrix(current_frame_, next_frame_, blendweight_frame);
            }
        }
    }

    int Navigation::getNextScene(const int base_frame, Direction direction) const{
        if(getNumFrames() == 1)
            return -1;
        //search optimal next frame by ray tracing

        //toy implementation
        if(direction == MOVE_FORWARD && base_frame < kNumFrames - 1){
            return base_frame + 1;
        }
        if(direction == MOVE_BACKWARD && base_frame > 0){
            return base_frame - 1;
        }



//        Vector3d ray_dir;
//        if(direction == MOVE_FORWARD){
//            ray_dir = extrinsics_[base_frame] * Vector4d(0.0, 0.0, 1.0, 0.0);
//        }else if(direction == MOVE_BACKWARD){
//            ray_dir = extrinsics_[base_frame] * Vector4d(0.0, 0.0, -1.0, 0.0);
//        }else if(direction == MOVE_LEFT){
//            ray_dir = extrinsics_[base_frame] * Vector4d(-1.0, 0.0, 0.0, 0.0);
//        }else if(direction == MOVE_LEFT){
//            ray_dir = extrinsics_[base_frame] * Vector4d(1.0, 0.0, 0.0, 0.0);
//        }
//
//        double best_angle = 0.0;
//        int best_frame = -1;
//        for(int i=0; i<camcenters_.size(); i++){
//            if(i == base_frame) {
//                continue;
//            }
//            Vector3d dir1 = camcenters_[i] - camcenters_[base_frame];
//            dir1.normalize();
//            double cur_angle = ray_dir.dot(dir1);
//            if(cur_angle > best_angle){
//                best_angle = cur_angle;
//                best_frame = i;
//            }
//        }
//        //printf("base_scene: %d, base_frame: %d best_scene: %d, angle: %.2f\n", base_frame.first, base_frame.second, best_scene, best_angle);
//        if(best_angle < 0.85)
//            return -1;
//        printf("best_frame:%d\n", best_frame);
//        return best_frame;
    }

    bool Navigation::MoveFrame(Direction direction){
        int temp_next = getNextScene(current_frame_,direction);
        if(temp_next) {
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

