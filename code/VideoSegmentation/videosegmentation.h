//
// Created by yanhang on 7/27/16.
//

#ifndef DYNAMICSTEREO_MERGESEGMENTATION_H
#define DYNAMICSTEREO_MERGESEGMENTATION_H

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "types.h"

namespace dynamic_stereo {
    namespace video_segment {

        struct VideoSegmentOption{
            explicit VideoSegmentOption(float threshold_):
                    threshold(threshold_), hasInvalid(false), refine(true),
                    smooth_size(3), theta(100), min_size(200), stride1(8), stride2(4),
                    pixel_feture_type(PixelFeature::PIXEL_VALUE), temporal_feature_type(TemporalFeature::TRANSITION_PATTERN){}
            float threshold;
            float hasInvalid;
            bool refine;
            int smooth_size;
            float theta;
            int min_size;
            int stride1;
            int stride2;
            PixelFeature pixel_feture_type;
            TemporalFeature temporal_feature_type;
        };

        void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output);

        int compressSegment(cv::Mat& segment);

        int segment_video(const std::vector<cv::Mat> &input, cv::Mat &output, const VideoSegmentOption& option);

        //Multi-frame multi-label grabcut algorithm
        //mask: for both input and output, with type CV_32SC1, the number indicates label id
        void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask,
                       const bool hasInvalid_ = false, const int iterCount = 3);

        cv::Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask);

        cv::Mat visualizeSegmentation(const cv::Mat &input);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
