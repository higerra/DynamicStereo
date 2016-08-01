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
                          const int smoothSize, const float c, const float theta, const int min_size,
                          const PixelFeature pfType,
                          const TemporalFeature tftype);

        cv::Mat visualizeSegmentation(const cv::Mat &input);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
