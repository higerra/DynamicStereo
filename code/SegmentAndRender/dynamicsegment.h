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

	void computeFrequencyConfidence(const std::vector<cv::Mat>& input, const float threshold, Depth& result, cv::Mat& frq_range);

	void segmentFlashy(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input,
					   std::vector<std::vector<Eigen::Vector2i> >& segments_flashy,
					   std::vector<Eigen::Vector2i>& ranges);

	void segmentDisplay(const FileIO& file_io, const int anchor, const std::vector<cv::Mat>& input,
						const std::string& classifierPath, const std::string& codebookPath, cv::Mat& result);

	void groupPixel(const cv::Mat& labels, std::vector<std::vector<Eigen::Vector2i> >& segments);

}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DYNAMICSEGMENT_H
