//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_REGIONDESCRIPTOR_H
#define DYNAMICSTEREO_REGIONDESCRIPTOR_H

#include "descriptor.h"
#include "../external/line_util/line_util.h"
#include "extracfeature.h"

namespace dynamic_stereo {
	namespace Feature {
		void computeFeatures(const std::vector<cv::Mat> &images,
							 const std::vector<cv::Point> &locs,
							 std::vector<double> &feature);
	}//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_REGIONDESCRIPTOR_H
