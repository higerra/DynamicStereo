#ifndef CAMERA_UTLI_H
#define CAMERA_UTLI_H

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include "camera.h"

namespace dynamic_rendering{
    namespace camera_utility{
	Eigen::Matrix4d inverseExtrinsic(const Eigen::Matrix4d& mat);

	inline Eigen::Matrix3d computeFundamental(const Camera& cam1,
						  const Camera& cam2){
	    Eigen::Matrix4d K = cam1.getIntrinsic();
	    Eigen::Vector4d center1_homo = cam1.getExtrinsic().block<4,1>(0,3);
	    Eigen::Vector4d epipole2_homo = K * cam2.getExtrinsic().inverse() * center1_homo;
	    Eigen::Vector3d epipole2(epipole2_homo[0], epipole2_homo[1], epipole2_homo[2]);
	    Eigen::Matrix4d P_homo = K * cam2.getExtrinsic().inverse() * cam1.getExtrinsic() * K.inverse();
	    Eigen::Matrix3d P = P_homo.block<3,3>(0,0);
	    Eigen::Matrix3d e2x;
	    e2x << 0, -1*epipole2[2], epipole2[1],
		epipole2[2], 0, -1*epipole2[0],
		-1*epipole2[1], epipole2[0], 0;
	    return e2x * P;
	}

	void triangulation(const std::vector<Eigen::Vector2d>& pt,
			   const std::vector<Camera>& cam,
			   Eigen::Vector3d& spacePt,
			   double& residual,
			   const bool min_repojection_error = true);
    }//namespace camera_utility
}//namespace dynamic_rendering

#endif
