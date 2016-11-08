#include "navigation.h"
#include "../base/file_io.h"
#include "../base/plane3D.h"
#include <iostream>
#include <fstream>
#include <string>
#define PI 3.1415927
using namespace std;
using namespace Eigen;

namespace dynamic_stereo{

const int Navigation::cx = 639.489;
const int Navigation::cy = 350.282;
const int Navigation::animation_stride = 15;
const int Navigation::animation_blendNum = 24;
const int Navigation::video_rate = 6;
const double Navigation::kbspeed = 2.0;
const int Navigation::speed_move_scene = 2;

Navigation::Navigation(const string& path):
    fov(35.0),
    cameraStatus(STATIC),
    current_frame(pair<int,int>(0,0)),
    next_frame(pair<int,int>(0,0)),
    animation_counter(0),
    video_counter(0),
    kbprogress(0.0),
    kbstride(0.02),
    kbdirection(1.0),
    blendweight_frame(1.0),
    blendweight_dynamic(1.0){

    //read scene configure file
    char buffer[1024];
    sprintf(buffer, "%s/scene_config.txt", path.c_str());
    ifstream scene_in(buffer);
    if(!scene_in.is_open()){
        cerr << "Cannot open scene_config.txt!";
        exit(-1);
    }
    scene_in >> kNumScenes;
    lookAtTable.resize(kNumScenes);
    cameras.resize(kNumScenes);
    walkablepath.resize(kNumScenes);
    scene_transform.resize(kNumScenes);
    scene_centers.resize(kNumScenes);
    vdirs.resize(kNumScenes);
    updirs.resize(kNumScenes);
    camcenters.resize(kNumScenes);

    Vector2d centerpix(cx, cy);
    string tempstring;
    //read scene transformation
    for(int i=0; i<kNumScenes; i++){
        scene_in >> tempstring;
        Camera temp;
        scene_in >> temp;
        scene_transform[i] = temp.getExtrinsic();
        walkablepath[i].resize(kNumScenes);
        for(int j=0; j<walkablepath[i].size(); j++)
            walkablepath[i][j] = 0;
    }
    //read path
    int kNumPath;
    scene_in >> kNumPath;
    for(int i=0; i<kNumPath; i++){
        int s,t;
        scene_in >> s >> t;
        walkablepath[s][t] = 1;
        walkablepath[t][s] = 1;
    }
    scene_in.close();

    for(int i=0; i<kNumScenes; i++){
        sprintf(buffer, "%s/scene%03d/", path.c_str(), i);
        scene_centers[i] = Vector3d(0,0,0);
        shared_ptr<FileIO> file_io(new FileIO(string(buffer)));
        int framenum = file_io->getTotalNum();
        if(framenum == 0){
            cerr << "Scene "<<i<<" is empty!"<<endl;
            exit(-1);
        }

        for(int j=0; j<framenum; j++){
            Camera curcam;
            ifstream camin(file_io->getOptimizedPose(j).c_str());
            if(!camin.is_open())
                camin.open(file_io->getPose(j).c_str());
            if(!camin.is_open()){
                cerr << "Navigation::constructor: cannot open camera file scene "<<i<<" frame "<<j<<endl;
                exit(-1);
            }
            camin >> curcam;
            curcam.setIntrinsic(1045.67, 1045.63, 639.489, 350.282);
            curcam.setExtrinsic(scene_transform[i] * curcam.getExtrinsic());
            scene_centers[i] += curcam.getCameraCenter();
            cameras[i].push_back(curcam);

            camcenters[i].push_back(curcam.getCameraCenter());
            Vector3d vdir = curcam.backProject(centerpix, 10.0) - camcenters[i].back();
            //Vector4d vdir_homo = curcam.getExtrinsic() * Vector4d(0.0,0.0,1.0,1.0);
            //Vector3d vdir = vdir_homo.block<3,1>(0,0);
            vdir.normalize();
            vdirs[i].push_back(vdir);
            Vector4d updir_homo = curcam.getExtrinsic() * Vector4d(0.0,-1.0,0.0,1.0);
            Vector3d updir = updir_homo.block<3,1>(0,0) - camcenters[i].back();
            updir.normalize();
            updirs[i].push_back(updir);

        }
        scene_centers[i] /= static_cast<double>(framenum);

        lookAtTable[i].resize(framenum);
        for(int j=0; j<framenum; j++)
            getLookAtParam(i, j, lookAtTable[i][j]);
    }

    //set initial camera
    projectionMatrix.setToIdentity();
    projectionMatrix.perspective(fov, (float)cx / (float) cy, 0.0, 100);
    renderCamera = getModelViewMatrix(current_frame, next_frame, blendweight_frame);
}

Navigation::~Navigation(){

}

void Navigation::getLookAtParam(const size_t sceneid, const size_t frameid, vector<float>&param){
    const Camera& curcam = cameras[sceneid][frameid];
    const Matrix4d& curext = curcam.getExtrinsic();
    Vector2d centerpix(cx, cy);
    Vector4d updir_homo = curext * Vector4d(0.0,-1.0,0.0,1.0);
    Vector3d cameracenter = curcam.getCameraCenter();
    Vector3d framecenter = curcam.backProject(centerpix, 10.0);
    param.resize(9);
    for(int i=0; i<3; i++)
        param[i] = static_cast<float>(cameracenter[i]);
    for(int i=3; i<6; i++)
        param[i] = static_cast<float>(framecenter[i-3]);
    for(int i=6; i<9; i++)
        param[i] = static_cast<float>(updir_homo[i-6] - cameracenter[i-6]);
}

QMatrix4x4 Navigation::getModelViewMatrix(const size_t sceneid1, const size_t frameid1,
                                          const size_t sceneid2, const size_t frameid2,
                                          const double percent){
    if(sceneid1 >= cameras.size() || sceneid2 >= cameras.size()){
        cerr << "Navigation::getmodelViewMatrix(): Scene id out of bound!"<<endl;
        exit(-1);
    }
    if(frameid1 >= cameras[sceneid1].size() || frameid2 >= cameras[sceneid2].size()){
        cerr << "Navigation::getmodelViewMatrix(): Frame id out of bound!"<<endl;
        exit(-1);
    }
    //interpolate the viewing direction
    vector<float> param(9);
    const vector<float>& t1 = lookAtTable[sceneid1][frameid1];
    const vector<float>& t2 = lookAtTable[sceneid2][frameid2];
    for(int i=0; i<9; i++){
        param[i] = percent * t1[i] + (1-percent) * t2[i];
    }
    const Vector3d& center1 = camcenters[sceneid1][frameid1];
    const Vector3d& center2 = camcenters[sceneid2][frameid2];

    const Vector3d& updir1 = updirs[sceneid1][frameid1];
    const Vector3d& updir2 = updirs[sceneid2][frameid2];
    const Vector3d& vdir1 = vdirs[sceneid1][frameid1];
    const Vector3d& vdir2 = vdirs[sceneid2][frameid2];

    Vector3d camcenter = percent * center1 + (1-percent) * center2;
    Vector3d updir = interpolateVector3D(updir1, updir2, 1 - percent);
    Vector3d vdir = interpolateVector3D(vdir1, vdir2, 1 - percent);
    Vector3d framecenter = vdir + camcenter;

    QMatrix4x4 m;
    m.setToIdentity();
    m.lookAt(QVector3D(camcenter[0], camcenter[1], camcenter[2]),
            QVector3D(framecenter[0], framecenter[1], framecenter[2]),
            QVector3D(updir[0], updir[1], updir[2]));
    //m.lookAt(QVector3D(param[0],param[1], param[2]),
    //        QVector3D(param[3],param[4], param[5]),
    //        QVector3D(param[6],param[7], param[8]));
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

void Navigation::updateNavigation(){
    if(cameraStatus == STATIC){
        blendweight_dynamic -= (double)video_rate / (double)animation_blendNum;
        if(blendweight_dynamic < 0)
            blendweight_dynamic = 1.0;

        //ken burns effect
        //pair<int,int> tempnextframe;
//        tempnextframe.first = current_frame.first;
//        if(kbprogress >= 0)
//            tempnextframe.second = (current_frame.second+animation_stride) % (int)cameras[current_frame.first].size();
//        else
//            tempnextframe.second = (current_frame.second+(int)cameras[current_frame.first].size() - animation_stride) % (int)cameras[current_frame.first].size();
//        renderCamera = getModelViewMatrix(current_frame, tempnextframe,
//                                          1.0 - kbspeed / (double)animation_blendNum * std::abs(kbprogress));
//        //printf("frame1:%d, frame2:%d kbprogress:%.4f, percent:%.4f\n", current_frame.second,
//        //       tempnextframe.second, kbprogress,1.0 / (double)animation_blendNum * std::abs(kbprogress));

//        kbprogress += kbdirection * kbstride;
//        if(std::abs(kbprogress) >= 1.0){
//            kbdirection *= -1.0;
//        }
    }else if(cameraStatus ==  TRANSITION_FRAME || cameraStatus == TRANSITION_SCENE){
        blendweight_dynamic -= (double)video_rate / (double)animation_blendNum;
        if(blendweight_dynamic < 0)
            blendweight_dynamic = 1.0;

        if(cameraStatus == TRANSITION_FRAME)
            animation_counter++;
        else
            animation_counter+=speed_move_scene;
        if(animation_counter >= animation_blendNum-1){
            cameraStatus = STATIC;
            animation_counter = 0;
            blendweight_frame = 1.0;
            current_frame = next_frame;
            renderCamera = getModelViewMatrix(current_frame, next_frame, blendweight_frame);
        }else{
            if(cameraStatus == TRANSITION_FRAME)
                blendweight_frame = animation::getFramePercent(animation_counter, animation_blendNum);
            else
                blendweight_frame = animation::getFramePercent(animation_counter, animation_blendNum);
            if(blendweight_frame < 0.001)
                printf("blendweight_frame:%.4f\n", blendweight_frame);
            renderCamera = getModelViewMatrix(current_frame, next_frame, blendweight_frame);
        }
    }
}

pair<int,int> Navigation::getNextScene(const pair<int,int>& base_frame, int direction) const{
    if(getNumScenes() == 1)
        return pair<int,int>(-1,-1);
    //compute verticle view plane
    const Camera& basecam = getCamera(base_frame.first, base_frame.second);
    Vector4d vdir_homo = basecam.getExtrinsic() * Vector4d(0.0,0.0,1.0,0.0);
    Vector3d vdir(vdir_homo[0], vdir_homo[1], vdir_homo[2]);
    //first select next scene, then select frame
    double best_angle = 0.0;
    int best_scene = -1;
    for(int i=0; i<scene_centers.size(); i++){
        //printf("----------------------------------\n");
        //printf("scene %d, center:(%.2f,%.2f,%.2f)\n", i, scene_centers[i][0], scene_centers[i][1], scene_centers[i][2]);
        if(i == base_frame.first)
            continue;
        if(walkablepath[base_frame.first][i] == 0){
            continue;
        }
        const Vector3d& curcenter = scene_centers[i];
        Vector3d dir1 = (curcenter - basecam.getCameraCenter());
        dir1.normalize();

        double cur_angle = vdir.dot(dir1) * (double)direction;
        //printf("curcenter:(%.2f,%.2f,%.2f)\n", curcenter[0], curcenter[1], curcenter[2]);
        //printf("vdir:(%.2f,%.2f,%.2f) dir1:(%.2f,%.2f,%.2f), angle:%.2f\n", vdir[0], vdir[1], vdir[2], dir1[0], dir1[1], dir1[2], cur_angle);
        if(cur_angle > best_angle){
            best_angle = cur_angle;
            best_scene = i;
        }
    }
    //printf("base_scene: %d, base_frame: %d best_scene: %d, angle: %.2f\n", base_frame.first, base_frame.second, best_scene, best_angle);
    if(best_angle < 0.85)
        return pair<int,int>(-1,-1);
    int best_frame = -1;
    best_angle = 0.0;
    for(int i=0; i<cameras[best_scene].size(); i++){
        Vector4d vdir2_homo = cameras[best_scene][i].getExtrinsic() * Vector4d(0.0,0.0,1.0,0.0);
        Vector3d vdir2 = vdir2_homo.block<3,1>(0,0);
        double cur_angle = vdir.dot(vdir2);
        if(cur_angle > best_angle){
            best_angle = cur_angle;
            best_frame = i;
        }
    }
    printf("best_scene:%d, best_frame:%d\n", best_scene, best_frame);
    return pair<int,int>(best_scene, best_frame);
}

void Navigation::rotate(int stride, int direction){
    current_frame = next_frame;
    int framenum = (int)cameras[next_frame.first].size();
    next_frame.second += direction * stride;
    next_frame.second = (next_frame.second+framenum) % framenum;
    animation_counter = 0;
    kbprogress = 0.0;
    kbdirection = 1.0;
}

bool Navigation::moveScene(int direction){
    pair<int,int> temp_next = getNextScene(current_frame,direction);
    if(temp_next.first == -1 || temp_next.second == -1)
        return false;
    current_frame = next_frame;
    next_frame = temp_next;
    animation_counter = 0;
    return true;
}

void Navigation::processKeyEvent(QKeyEvent* e){
    if(cameraStatus != STATIC)
        return;
    switch(e->key()){
    case Qt::Key_Left:
        rotate(animation_stride, -1);
        cameraStatus = TRANSITION_FRAME;
        break;
    case Qt::Key_Right:
        rotate(animation_stride, 1);
        cameraStatus = TRANSITION_FRAME;
        break;
    case Qt::Key_Up:{
        if(moveScene(1))
            cameraStatus = TRANSITION_SCENE;
        break;
    }
    case Qt::Key_Down:{
        if(moveScene(-1))
            cameraStatus = TRANSITION_SCENE;
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
    case(MOVE):{
        int stride = animation_stride;
        int direction = dx < 0 ? 1 : -1;
        if(dx < 0) dx = -1 * dx;
        if(dx < 10) return;
        if(dx < 50)
            stride -= 10;
        if(dx > 200)
            stride += 10;
        rotate(stride,direction);
        cameraStatus = TRANSITION_FRAME;
        break;
    }
    case(CLICK):
        break;
    case(DOUBLE_CLICK):{
        if(moveScene(1))
            cameraStatus = TRANSITION_SCENE;
        break;
    }
    default:
        break;
    }
}

}//namespace dynamic_rendering

