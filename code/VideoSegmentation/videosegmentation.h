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

        int compressSegment(cv::Mat& segment);

        int segment_video(const VideoMat &input, cv::Mat &output,
                          const float c,  const bool hasInvalid = false, const bool refine = true,
                          const int smoothSize = 9, const float theta = 100, const int min_size = 200,
                          const int stride1 = 8, const int stride2 = 4,
                          const PixelFeature pfType = PixelFeature::PIXEL,
                          const TemporalFeature tftype = TemporalFeature::TRANSITION_PATTERN);

        //Multi-frame multi-label grabcut algorithm
        //mask: for both input and output, with type CV_32SC1, the number indicates label id
        void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask,
                       const bool hasInvalid_ = false, const int iterCount = 3);

        cv::Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask);

        cv::Mat visualizeSegmentation(const cv::Mat &input);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
