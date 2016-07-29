//
// Created by yanhang on 7/27/16.
//

#ifndef DYNAMICSTEREO_MERGESEGMENTATION_H
#define DYNAMICSTEREO_MERGESEGMENTATION_H

#include <string>
#include <memory>

#include "distance_metric.h"
#include "pixel_feature.h"

namespace dynamic_stereo {

    void edgeAggregation(const VideoMat &input, cv::Mat &output);

    int segment_video(const VideoMat& input, cv::Mat& output,
                      const int smoothSize, const float c, const float theta, const int min_size);

    cv::Mat visualizeSegmentation(const cv::Mat& input);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
