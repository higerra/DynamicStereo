//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_REGIONDESCRIPTOR_H
#define DYNAMICSTEREO_REGIONDESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <string>
#include <glog/logging.h>
#include "../external/line_util/line_util.h"

namespace dynamic_stereo {
	namespace Feature {
		struct Region {
			Region(const std::vector<cv::Point> &locs_, const cv::Mat &img_);

			float width;
			float height;

			float area;
			int kEdge;
			float aspectRatio;
			float area_chall;
			Eigen::Vector4f bbox;
			std::vector<float> hist_diff;
			std::vector<float> hist_color;

			const int kBin;

			std::vector<cv::Point> locs;
			std::vector<Eigen::Vector3f> pixs;
			const cv::Mat &img;

			void computeFeatures();
		};

		class RegionDescriptor {
		public:

		};
	}//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_REGIONDESCRIPTOR_H
