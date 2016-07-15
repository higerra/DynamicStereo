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

	double testForest(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::DTrees> forest);
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_RANDOMFOREST_H
