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
                          const int stride1 = 8, const int stride2 = 4,
                          const PixelFeature pfType = PixelFeature::PIXEL,
                          const TemporalFeature tftype = TemporalFeature::TRANSITION_PATTERN);

        //multi frame grab cut
        //mask: for both input and output
        void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask, const int iterCount = 10);

        cv::Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask);

        //joint refine the boundary of all segments based on appearance
        cv::Mat segmentRefinement(const std::vector<cv::Mat>& images, const cv::Mat& segments, const float marginRatio = 0.1);

        cv::Mat visualizeSegmentation(const cv::Mat &input);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
