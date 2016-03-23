//
// Created by Yan Hang on 3/22/16.
//

#ifndef DYNAMICSTEREO_MODEL_H
#define DYNAMICSTEREO_MODEL_H

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <vector>

namespace dynamic_stereo{

	template<typename T>
	struct StereoModel{
		StereoModel(const cv::Mat& img_, const int nLabel_, const double MRFRatio_, const double ws_):
				image(img_.clone()), width(img_.cols), height(img_.rows), nLabel(nLabel_), MRFRatio(MRFRatio_), weight_smooth(ws_){
			CHECK(image.data) << "Empty image";
		}
		void allocate(){
			unary.resize(width * height * nLabel);
			vCue.resize(width * height);
			hCue.resize(width * height);
		}

		std::vector<T> unary;
		std::vector<T> vCue;
		std::vector<T> hCue;
		const int nLabel;
		const int width;
		const int height;
		const double MRFRatio;

		//////////////////////////////////
		//NOTICE!!!! weigth_smooth is in original scale. Multiply with MRFRatio when needed
		const double weight_smooth;

		const cv::Mat image;

		double min_disp;
		double max_disp;

		const T operator()(int id, int label)const{
			CHECK_LT(id, width * height);
			CHECK_LT(label, nLabel);
			return unary[id * nLabel + label];
		}
		T& operator()(int id, int label){
			CHECK_LT(id, width * height);
			CHECK_LT(label, nLabel);
			return unary[id * nLabel + label];
		}

		inline double dispToDepth(const double d) const{
			return 1.0/(min_disp + d * (max_disp - min_disp) / (double) nLabel);
		}
		inline double depthToDisp(double depth) const{
			return (1.0 / depth * (double)nLabel - min_disp)/ (max_disp - min_disp);
		}
	};
}
#endif //DYNAMICSTEREO_MODEL_H
