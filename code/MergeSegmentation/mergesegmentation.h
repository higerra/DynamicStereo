//
// Created by yanhang on 7/27/16.
//

#ifndef DYNAMICSTEREO_MERGESEGMENTATION_H
#define DYNAMICSTEREO_MERGESEGMENTATION_H

#include <vector>
#include <string>
#include <memory>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "../external/segment_gb/segment-image.h"

namespace dynamic_stereo {

    void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
