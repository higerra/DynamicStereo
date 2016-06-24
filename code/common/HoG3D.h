//
// Created by Yan Hang on 6/21/16.
//

#ifndef DYNAMICSTEREO_HOG3D_H
#define DYNAMICSTEREO_HOG3D_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace dynamic_stereo {
	namespace Feature {
		class HoG3D {
		private:
			Eigen::MatrixXd PMat;
		};
	}//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_HOG3D_H
