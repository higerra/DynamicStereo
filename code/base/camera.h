#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <vector>

namespace dynamic_rendering{

    class FileIO;

    class Camera{
    public:
        Camera(): extrinsic(Eigen::Matrix4d::Identity()),k1(0),k2(0),k3(0){}
        Camera(const FileIO& file_io, int id);

        void initialize(const FileIO& fil_io, int id);

        inline const Eigen::Matrix4d& getIntrinsic() const {return intrinsic;}
        inline const Eigen::Matrix4d& getExtrinsic() const{ return extrinsic;}
        inline const Eigen::Matrix4d& getProjection() const {return intrinsic * extrinsic.inverse();}

        inline void setIntrinsic(double fx, double fy, double cx, double cy, double k1_=0, double k2_=0, double k3_=0){
            intrinsic << fx,0,cx,0,
                    0,fy,cy,0,
                    0,0,1,0,
                    0,0,0,1;
            k1 = k1_;
            k2 = k2_;
            k3 = k3_;
            isIntrinsic_set = true;
        }
        inline void setIntrinsic(const Eigen::Matrix4d& intrinsic_){
            intrinsic = intrinsic_;
        }

        inline bool isIntrinsicSet() const{return isIntrinsic_set;}

        inline void setExtrinsic(const Eigen::Matrix4d &extrinsic_){
            extrinsic = extrinsic_;
        }

        Eigen::Vector2d projectToImage(const Eigen::Vector3d &src)const;
        Eigen::Vector3d backProject(const Eigen::Vector2d &src, double distance)const;

        inline Eigen::Vector3d transformToLocal(const Eigen::Vector3d &src)const{
            Eigen::Vector4d temp(src[0], src[1], src[2],1.0);
            Eigen::Vector4d localpt_homo = getExtrinsic().inverse() * temp;
            return Eigen::Vector3d(localpt_homo[0], localpt_homo[1], localpt_homo[2]);
        }

        inline Eigen::Vector3d getCameraCenter() const{
            return Eigen::Vector3d(extrinsic(0,3), extrinsic(1,3), extrinsic(2,3));
        }

        friend std::ifstream& operator>>(std::ifstream& istr, Camera& camera);
        friend std::ofstream& operator<<(std::ofstream& ostr, const Camera& camera);

    private:
        static bool isIntrinsic_set;
        static Eigen::Matrix4d intrinsic;
        Eigen::Matrix4d extrinsic;
        double k1, k2, k3;
    };


} //namespace dynamic_rendering

#endif



















