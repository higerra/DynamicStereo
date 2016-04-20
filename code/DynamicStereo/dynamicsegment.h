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
        DynamicSegment(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_,
                       const std::vector<Depth>& depths_, const std::vector<int>& depthInd_);

	    //void getGeometryConfidence(Depth& geoConf) const;

	    void segment(const std::vector<cv::Mat>& warppedImg, cv::Mat& result) const;
    private:
		void computeFrequencyConfidence(const std::vector<cv::Mat>& warppedImg, Depth& result) const;
        const FileIO& file_io;
        SfMModel sfmModel;

        const int anchor;
	    int offset;
        const int downsample;
    };
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
