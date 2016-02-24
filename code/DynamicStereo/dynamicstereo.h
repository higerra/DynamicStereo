//
// Created by yanhang on 2/24/16.
//

#ifndef DYNAMICSTEREO_DYNAMICSTEREO_H
#define DYNAMICSTEREO_DYNAMICSTEREO_H
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <theia/theia.h>
#include "base/utility.h"
#include "base/configurator.h"
#include "base/depth.h"
#include "base/file_io.h"
#include "MRF2.2/mrf.h"
#include "MRF2.2/GCoptimization.h"

namespace dynamic_stereo {
    class DynamicStereo {
    public:
        DynamicStereo(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_);
        void verifyEpipolarGeometry(const int id1, const int id2,
                                                   const Eigen::Vector2d& pt,
                                                   cv::Mat &imgL, cv::Mat &imgR);
        void runStereo();

        inline int getAnchor()const{return anchor;}
        inline int gettWindow() const {return tWindow;}
        inline int getOffset() const {return offset;};
    private:
        void initMRF();
        void computeMinMaxDepth();
        void assignDataTerm();
        std::shared_ptr<MRF> createProblem();

        const FileIO& file_io;
        const int anchor;
        const int tWindow;
        const int downsample;
        int offset;

        int width;
        int height;

        const int depthResolution;
        const int pR; //radius of patch
        double min_disp;
        double max_disp;

        //downsampled version
        std::vector<cv::Mat> images;
        theia::Reconstruction reconstruction;
        Depth refDepth;

        //for MRF
        std::vector<MRF::CostVal> MRF_data;
        std::vector<MRF::CostVal> MRF_smooth;
    };
}

#endif //DYNAMICSTEREO_DYNAMICSTEREO_H
