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
    void splitSamples(const cv::Mat& input, std::vector<cv::Mat>& output, const int kFold);

    void train(const std::string& input_path, const std::string& output_path, const std::string& type);
    cv::Mat predict(const std::string& model_path, const std::string& data_path, const int width, const int height, const std::string& type);

    void trainSVMWithPlatt(const std::string& input_path, const std::string& output_path);

    void trainPlattScaling(const cv::Mat& data, const std::string& output_path);
    void predictPlattScaling(const std::string& model_path, const cv::Mat& data, cv::Mat& result);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAIN_H
