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

    void train(const std::string& input_path, const std::string& output_path, const std::string& type);
    cv::Mat predict(const std::string& model_path, const std::string& data_path, const int width, const int height, const std::string& type);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAIN_H