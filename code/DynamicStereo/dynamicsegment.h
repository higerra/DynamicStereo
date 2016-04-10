//
// Created by yanhang on 4/10/16.
//

#ifndef DYNAMICSTEREO_DYNAMICSEGMENT_H
#define DYNAMICSTEREO_DYNAMICSEGMENT_H
#include <iostream>
#include <string>
#include "../base/file_io.h"
#include "../base/depth.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <theia/theia.h>
#include <glog/logging.h>
#include "model.h"
namespace dynamic_stereo {
    class DynamicSegment {
    public:
        DynamicSegment(const FileIO& file_io_, const int anchor_, const int downsample_,
                       const std::vector<Depth>& depths_, const std::vector<int>& depthInd_);
    private:
        const FileIO& file_io;
        std::vector<cv::Mat> images;
        const std::vector<Depth>& depths;
        const std::vector<int>& depthInd;
        SfMModel sfmModel;

        const int anchor;
        const int downsample;
    };
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
