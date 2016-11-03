//
// Created by yanhang on 7/27/16.
//

#ifndef DYNAMICSTEREO_MERGESEGMENTATION_H
#define DYNAMICSTEREO_MERGESEGMENTATION_H

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../external/segment_gb/segment-image.h"

#include "types.h"

namespace dynamic_stereo {
    namespace video_segment {

        struct Region;

        struct VideoSegmentOption{
            explicit VideoSegmentOption(float threshold_):
                    threshold(threshold_), hasInvalid(false), refine(true),
                    smooth_size(3), theta(100), min_size(50), stride1(8), stride2(4), w_appearance(0.001),
                    pixel_feture_type(PixelFeature::PIXEL_VALUE), temporal_feature_type(TemporalFeature::TRANSITION_PATTERN),
                    hier_iter(-1), region_temporal_feature_type(TemporalFeature::COMBINED){
                w_transition = 1 - w_appearance;
            }
            float threshold;
            float hasInvalid;
            bool refine;
            int smooth_size;
            float theta;
            int min_size;
            int stride1;
            int stride2;
            float w_appearance;
            float w_transition;
            PixelFeature pixel_feture_type;
            TemporalFeature temporal_feature_type;

            int hier_iter;
            TemporalFeature region_temporal_feature_type;
        };

        void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output);

        int compressSegment(cv::Mat& segment);

        int segment_video(const std::vector<cv::Mat> &input, cv::Mat &output, const VideoSegmentOption& option);

        void BuildEdgeMap(const std::vector<Region*>& regions, std::vector<segment_gb::edge> & edge_map,
                          const int width, const int height);

        int HierarchicalSegmentation(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const VideoSegmentOption& option);

        //Multi-frame multi-label grabcut algorithm
        //mask: for both input and output, with type CV_32SC1, the number indicates label id
        void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask,
                       const bool hasInvalid_ = false, const int iterCount = 3);

        cv::Mat localRefinement(const std::vector<cv::Mat>& images, const int R_erode, const int R_dilate, const int min_area, cv::Mat& mask);

        cv::Mat visualizeSegmentation(const cv::Mat &input);

        bool LoadHierarchicalSegmentation(const std::string& filename, std::vector<cv::Mat>& segments);

        void SaveHierarchicalSegmentation(const std::string& filename, const std::vector<cv::Mat>& segments);
    }//namespace video_segment
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MERGESEGMENTATION_H
