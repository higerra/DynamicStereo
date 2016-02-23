#include "camera.h"
#include "file_io.h"
using namespace Eigen;
using namespace std;

namespace dynamic_rendering{
    Camera::Camera(const FileIO& file_io, int id){
        initialize(file_io, id);
    }

    void Camera::initialize(const FileIO& file_io, int id){
        assert(id < file_io.getTotalNum());
        ifstream posein(file_io.getOptimizedPose(id).c_str());
	if(!posein.is_open()){
	    posein.close();
	    posein.open(file_io.getPose(id).c_str());
	}
	if(!posein.is_open()){
	    printf("%s\n", file_io.getOptimizedPose(id).c_str());
	    cerr << "Camera::initialize(): Cannot open pose file!"<<endl;
	    exit(-1);
	}
        posein >> (*this);
    }

    ifstream& operator >> (ifstream& istr, Camera& camera){
        for(int y=0; y<4; y++){
            for(int x=0; x<4; x++)
                istr >> camera.extrinsic(y,x);
        }
        return istr;
    }

    ofstream& operator << (ofstream& ostr, const Camera& camera){
        for(int y=0; y<4; y++){
            for(int x=0; x<4; x++)
                ostr << camera.extrinsic(y,x)<<' ';
            ostr << endl;
        }
        return ostr;
    }

    bool Camera::isIntrinsic_set = false;
    Matrix4d Camera::intrinsic = Matrix4d::Identity();

    Eigen::Vector2d Camera::projectToImage(const Eigen::Vector3d &src)const{
        //currently doesn't correct distortion
        assert(isIntrinsic_set);
        Vector4d pt_undis = getProjection() * Vector4d(src[0], src[1], src[2], 1.0);
        double x; double y;
        const double cx = intrinsic(0,2);
        const double cy = intrinsic(1,2);
        if(pt_undis[2] != 0){
            x = pt_undis[0] / pt_undis[2];
            y = pt_undis[1] / pt_undis[2];
        }
        double r2 = (x-cx) * (y-cy);
        double r4 = r2*r2;
        double r6 = r2*r4;
        // x = x * (1 + k1*r2 + k2*r4 + k3*r6);
        // y = y * (1 + k1*r2 + k2*r4 + k3*r6);
        return Vector2d(x,y);
    }

    Eigen::Vector3d Camera::backProject(const Eigen::Vector2d& src, double distance)const{
        //current doesn't correct distortion
        Vector4d worldpt_homo = extrinsic * intrinsic.inverse() * Vector4d(distance* src[0], distance* src[1], distance, 1.0);
        Vector3d worldpt(worldpt_homo[0], worldpt_homo[1], worldpt_homo[2]);;
        if(worldpt_homo[3]!=0){
            worldpt[0] /= worldpt_homo[3];
            worldpt[1] /= worldpt_homo[3];
            worldpt[2] /= worldpt_homo[3];
        }
        return worldpt;
    }
}//namespace dynamic_rendering
