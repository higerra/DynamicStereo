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
    class DynamicSegment{
    public:
        DynamicSegment(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_,
                       const std::vector<Depth>& depths_, const std::vector<int>& depthInd_);

	    //void getGeometryConfidence(Depth& geoConf) const;

	    void segmentFlashy(const std::vector<cv::Mat>& input, cv::Mat& result) const;
		void segmentDisplay(const std::vector<cv::Mat>& input, const cv::Mat& segnetMask, cv::Mat& displayLabels,
							std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay) const;

    private:
		void computeColorConfidence(const std::vector<cv::Mat>& input, Depth& result) const;
		void computeFrequencyConfidence(const std::vector<cv::Mat>& input, Depth& result) const;

		//compute threshold for nlog
		double computeNlogThreshold(const std::vector<cv::Mat>& input, const cv::Mat& inputMask, const int K) const;

		void getHistogram(const std::vector<cv::Vec3b>& samples, std::vector<double>& hist, const int nBin) const;
	    void assignColorTerm(const std::vector<cv::Mat>& warped, const cv::Ptr<cv::ml::EM> fgModel, const cv::Ptr<cv::ml::EM> bgModel,
							 std::vector<double>& colorTerm) const;

//		void solveMRF(const std::vector<double>& unary,
//					  const std::vector<double>& vCue, const std::vector<double>& hCue,
//					  const cv::Mat& img, const double weight_smooth,
//					  cv::Mat& result) const;

        const FileIO& file_io;
        SfMModel sfmModel;

        const int anchor;
	    int offset;
        const int downsample;
    };
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
