//
// Created by yanhang on 3/4/16.
//

#ifndef PARALLELFUSION_LOCAL_MATCHER_H
#define PARALLELFUSION_LOCAL_MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <Eigen/Eigen>
#include "../base/utility.h"
#include "../base/depth.h"

namespace local_matcher {
    void samplePatch(const cv::Mat &img, const Eigen::Vector2d &loc, const int pR, std::vector<double> &pix);

    void getSSDArray(const std::vector<std::vector<double> > &patches, const int refId,
                     std::vector<double> &mCost);

    void getNCCArray(const std::vector<std::vector<double> >& patches, const int refId,
                     std::vector<double>& mCost);

    double medianMatchingCost(const std::vector<std::vector<double> >& patches, const int refId);

    double sumMatchingCost(const std::vector<std::vector<double> > &patches, const int refId);
}

#endif //PARALLELFUSION_LOCAL_MATCHER_H
