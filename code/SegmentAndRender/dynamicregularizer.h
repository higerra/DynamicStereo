//
// Created by yanhang on 4/29/16.
//

#ifndef DYNAMICSTEREO_DYNAMICREGULARIZER_H
#define DYNAMICSTEREO_DYNAMICREGULARIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <Eigen/Eigen>

namespace dynamic_stereo{

    class Depth;

    void dynamicRegularization(const std::vector<cv::Mat>& input,
                               const std::vector<std::vector<Eigen::Vector2d> >& segments,
                               std::vector<cv::Mat>& output, const double weight_smooth);

    void regularizationPoisson(const std::vector<cv::Mat>& input,
                               const std::vector<std::vector<Eigen::Vector2d> >& segments,
                               std::vector<cv::Mat>& output, const double ws, const double wt);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DYNAMICREGULARIZER_H
