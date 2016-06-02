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
//#include "../common/stereomodel.h"
#include "../common/descriptor.h"

namespace dynamic_stereo {

	void segmentFlashy(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input, cv::Mat& result);

	void segmentDisplay(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input, const cv::Mat& segnetMask,
	                    const std::string& classifierPath, cv::Mat& displayLabels,
	                    std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay);

	void filterBoudary(const std::vector<cv::Mat>& images, cv::Mat& input);

	void computeFrequencyConfidence(const std::vector<cv::Mat>& input, Depth& result);

	cv::Mat getClassificationResult(const std::vector<cv::Mat>& input,
	                                const std::shared_ptr<Feature::FeatureConstructor> descriptor, const cv::Ptr<cv::ml::StatModel> classifier,
	                                const int stride);

	//routine for importing video segmentation
	void importVideoSegmentation(const std::string& path, std::vector<cv::Mat>& video_segments);

}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
