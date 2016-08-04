//
// Created by yanhang on 7/27/16.
//

#ifndef DYNAMICSTEREO_MERGESEGMENTATION_H
#define DYNAMICSTEREO_MERGESEGMENTATION_H

#include <string>
#include <memory>

#include "pixel_feature.h"

namespace dynamic_stereo {

    namespace video_segment {
        void edgeAggregation(const VideoMat &input, cv::Mat &output);

        int segment_video(const VideoMat &input, cv::Mat &output,
                          const float c, const int smoothSize = 9, const float theta = 100, const int min_size = 200,
                          const PixelFeature pfType = PixelFeature::PIXEL,
                          const TemporalFeature tftype = TemporalFeature::TRANSITION_PATTERN);

        cv::Mat visualizeSegmentation(const cv::Mat &input);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
