//
// Created by yanhang on 5/19/16.
//

#ifndef DYNAMICSTEREO_TRAIN_H
#define DYNAMICSTEREO_TRAIN_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dynamic_stereo{

    void perturbSamples(cv::Mat& samples);
    void splitSamples(const cv::Ptr<cv::ml::TrainData> input, std::vector<cv::Mat>& outputSample, std::vector<cv::Mat>& outputLabel, int kFold);

    void train(const std::string& input_path, const std::string& output_path, const std::string& type);
    cv::Mat predict(const std::string& model_path, const std::string& data_path, const int width, const int height, const std::string& type);

    void trainSVMWithPlatt(const std::string& input_path, const std::string& output_path);
	cv::Mat predictSVMWithPlatt(const std::string& model_path, const std::string& data_path, const int width, const int height);

    cv::Ptr<cv::ml::LogisticRegression> trainPlattScaling(const cv::Mat& trainData);
    void predictPlattScaling(const std::string& model_path, const cv::Mat& data, cv::Mat& result);

	inline cv::Mat calc_sigmond(const cv::Mat& data) {
		cv::Mat dest;
		cv::exp(-data, dest);
		return 1.0/(1.0+dest);
	}
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAIN_H
