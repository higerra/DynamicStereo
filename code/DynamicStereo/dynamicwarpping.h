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
                        const int nLabel_, const std::vector<Depth>& depths, const std::vector<int>& depthind);

        void warpToAnchor(const cv::Mat& mask, std::vector<cv::Mat>& warpped, const bool earlyTerminate) const;

        inline int getWidth() const{return width;}
        inline int getHeight() const{return height;}
        inline int getOffset() const{return offset;}

	    inline double depthToDisp(const double d, const double min_depth, const double max_depth) const{
		    CHECK_GT(min_depth, 0.0);
		    CHECK_GT(max_depth, 0.0);
		    double min_disp = 1.0 / max_depth;
		    double max_disp = 1.0 / min_depth;
		    return (1.0 / d * (double)nLabel - min_disp)/ (max_disp - min_disp);
	    }
    private:
        void initZBuffer(const std::vector<Depth>& depths, const std::vector<int>& depthind);
        void updateZBuffer(const Depth& depth, Depth& zb, const theia::Camera& cam1, const theia::Camera& cam2) const;
        const FileIO& file_io;
        SfMModel sfmModel;
        Depth refDepth;
        const int anchor;
        const int downsample;
	    const int nLabel;
        int offset;
        int width;
        int height;
        std::vector<cv::Mat> images;
        std::vector<Depth> zBuffers;
	    std::vector<double> min_depths;
	    std::vector<double> max_depths;
    };
}

#endif //DYNAMICSTEREO_DYNAMICWARPPING_H
