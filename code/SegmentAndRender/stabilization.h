//
// Created by yanhang on 9/11/16.
//

#ifndef DYNAMICSTEREO_STABILIZATION_H
#define DYNAMICSTEREO_STABILIZATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "../base/utility.h"

namespace dynamic_stereo {

    enum StabAlg{
        GRID,
        FLOW,
        SUBSTAB,
        TRACK
    };

    void stabilizeSegments(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output,
                           const std::vector<std::vector<Eigen::Vector2i> >& segments,
                           const std::vector<Eigen::Vector2i>& ranges, const int anchor,
                           const double lambda, const StabAlg alg = FLOW);

//    void flowStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double lambda,
//                           const cv::InputArray inputMask = cv::noArray());
//
//    void gridStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double lambda,
//                           const int step = 1);
//
//    void trackStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double threshold, const int tWindow);

    int trackStabilizationGlobal(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                                 const double threshold, const int tWindow);


}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_STABILIZATION_H
