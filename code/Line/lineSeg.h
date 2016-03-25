//
// Created by yanhang on 3/24/16.
//

#ifndef DYNAMICSTEREO_LINESEG_H
#define DYNAMICSTEREO_LINESEG_H

#include <theia/theia.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <glog/logging.h>
#include <string>
#include "../base/file_io.h"
#include "../base/utility.h"

namespace dynamic_stereo {

    class LineSeg {
    public:
        LineSeg(const FileIO& file_io_);
        void undistort(const cv::Mat& input, cv::Mat& output, const theia::Camera& cam) const;
        void runLSD();
    private:
        typedef std::pair<int, theia::ViewId> IdPair;
        const FileIO& file_io;
        int width;
        int height;
        theia::Reconstruction reconstruction;
        std::vector<cv::Mat> images;
        std::vector<IdPair> orderedId;
    };
}

#endif //DYNAMICSTEREO_LINESEG_H
