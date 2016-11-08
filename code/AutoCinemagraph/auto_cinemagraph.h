//
// Created by yanhang on 11/7/16.
//

#ifndef DYNAMICSTEREO_AUTO_CINEMAGRAPH_H
#define DYNAMICSTEREO_AUTO_CINEMAGRAPH_H

#include <iostream>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
namespace dynamic_stereo {

    void ComputeOpticalFlow(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& opt_flow);

    void LoadVideo(const std::string& path, std::vector<cv::Mat>& images, int max_frame = -1);

    void GetPixelScore(const std::vector<cv::Mat>& opt_flow, std::vector<cv::Mat>& scores);

    void OptimalSpatialInterval(const std::vector<cv::Mat>& scores, std::vector<std::vector<int> >& spatial);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_AUTO_CINEMAGRAPH_H
