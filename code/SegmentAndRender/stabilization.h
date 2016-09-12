//
// Created by yanhang on 9/11/16.
//

#ifndef DYNAMICSTEREO_STABILIZATION_H
#define DYNAMICSTEREO_STABILIZATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "../base/utility.h"

namespace dynamic_stereo {

    struct WarpGrid{
        WarpGrid(const int width, const int height, const int gridW_, const int gridH_);
        std::vector<Eigen::Vector2d> gridLoc;
        int gridW, gridH;
        double blockW, blockH;
    };

    void getGridIndAndWeight(const WarpGrid& grid, const Eigen::Vector2d& pt, Eigen::Vector4i& ind, Eigen::Vector4d& w);

    void gridStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double ws, const int step = 1);


}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_STABILIZATION_H
