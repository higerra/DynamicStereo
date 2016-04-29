//
// Created by yanhang on 4/29/16.
//

#ifndef DYNAMICSTEREO_DYNAMICREGULARIZER_H
#define DYNAMICSTEREO_DYNAMICREGULARIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace dynamic_stereo{

    void dynamicRegularization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DYNAMICREGULARIZER_H
