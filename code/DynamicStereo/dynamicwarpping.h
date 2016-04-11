//
// Created by yanhang on 4/10/16.
//

#ifndef DYNAMICSTEREO_DYNAMICWARPPING_H
#define DYNAMICSTEREO_DYNAMICWARPPING_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "../base/depth.h"
#include "../base/file_io.h"
#include "dynamic_utility.h"

namespace dynamic_stereo {
    class DynamicWarpping {
    public:
        DynamicWarpping(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_,
                        const std::vector<Depth>& depths, const std::vector<int>& depthind);

        void warpToAnchor(const cv::Mat& mask, std::vector<cv::Mat>& warpped, const bool earlyTerminate) const;

        inline int getWidth() const{return width;}
        inline int getHeight() const{return height;}
        inline int getOffset() const{return offset;}
    private:
        void initZBuffer(const std::vector<Depth>& depths, const std::vector<int>& depthind);
        void updateZBuffer(const Depth& depth, Depth& zb, const theia::Camera& cam1, const theia::Camera& cam2) const;
        const FileIO& file_io;
        SfMModel sfmModel;
        Depth refDepth;
        const int anchor;
        const int downsample;
        int offset;
        int width;
        int height;
        std::vector<cv::Mat> images;
        std::vector<Depth> zBuffers;
    };
}

#endif //DYNAMICSTEREO_DYNAMICWARPPING_H
