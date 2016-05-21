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
#include "../extractFeature/descriptor.h"

namespace dynamic_stereo {

    class DynamicSegment{
    public:
        DynamicSegment(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_,
                       const std::vector<Depth>& depths_, const std::vector<int>& depthInd_);

	    //void getGeometryConfidence(Depth& geoConf) const;

	    void segmentFlashy(const std::vector<cv::Mat>& input, cv::Mat& result) const;
		void segmentDisplay(const std::vector<cv::Mat>& input, const cv::Mat& segnetMask, cv::Mat& displayLabels,
							std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay) const;

    private:
		void computeFrequencyConfidence(const std::vector<cv::Mat>& input, Depth& result) const;

		cv::Mat getClassificationResult(const std::vector<cv::Mat>& input,
										const std::shared_ptr<Feature::FeatureConstructor> descriptor, const cv::Ptr<cv::ml::StatModel> classifier,
										const int stride) const;

        const FileIO& file_io;
        SfMModel sfmModel;

        const int anchor;
	    int offset;
        const int downsample;
    };
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
