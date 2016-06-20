//
// Created by yanhang on 4/10/16.
//

#ifndef DYNAMICSTEREO_DYNAMICSEGMENT_H
#define DYNAMICSTEREO_DYNAMICSEGMENT_H
#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>

#include "../base/file_io.h"
#include "../base/depth.h"


namespace dynamic_stereo {

	namespace Feature{
		class FeatureConstructor;
	}

	void filterBoudary(const std::vector<cv::Mat>& videoSeg, cv::Mat& inputMask);
	void filterBySegnet(const std::vector<cv::Mat>& videoSeg, const cv::Mat& segMask, cv::Mat& inputMask);

	void computeFrequencyConfidence(const std::vector<cv::Mat>& input, Depth& result);

	cv::Mat getClassificationResult(const std::vector<cv::Mat>& input,
	                                const std::shared_ptr<Feature::FeatureConstructor> descriptor, const cv::Ptr<cv::ml::StatModel> classifier,
	                                const int stride);

	void importVideoSegmentation(const std::string& path, std::vector<cv::Mat>& video_segments);

	void segmentFlashy(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input, cv::Mat& result);

	void segmentDisplay(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input, const cv::Mat& segnetMask,
						const std::string& classifierPath, cv::Mat& result);

	void groupPixel(const cv::Mat& labels, std::vector<std::vector<Eigen::Vector2d> >& segments);

	cv::Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask);
	//multi frame grab cut
	//mask: for both input and output
	void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask, const int iterCount = 10);


}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
