//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_REGIONDESCRIPTOR_H
#define DYNAMICSTEREO_REGIONDESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <string>

namespace Feature {
    struct Region{
        float area;
        int kEdge;
        float aspectRatio;
        float area_chall;
        Eigen::Vector4f bbox;

        std::vector<cv::Point> locs;
        std::vector<cv::Vec3b> pixs;
        const cv::Mat& img;
    };

    class RegionDescriptor{
    public:

    };
}//namespace Feature

#endif //DYNAMICSTEREO_REGIONDESCRIPTOR_H
