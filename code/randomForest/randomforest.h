//
// Created by Yan Hang on 7/7/16.
//

#ifndef DYNAMICSTEREO_RANDOMFOREST_H
#define DYNAMICSTEREO_RANDOMFOREST_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "../common/regiondescriptor.h"

namespace dynamic_stereo {
	cv::Ptr<cv::ml::TrainData> convertTrainData(const Feature::TrainSet& trainset);

	void saveTrainData(const std::string& path, const Feature::TrainSet& trainset);

	void splitTrainSet(const Feature::TrainSet& trainset, Feature::TrainSet& set1, Feature::TrainSet& set2);

	void balanceTrainSet(Feature::TrainSet& trainset, Feature::TrainSet& unused, const double max_ratio);
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_RANDOMFOREST_H
