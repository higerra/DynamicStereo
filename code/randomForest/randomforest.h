//
// Created by Yan Hang on 7/7/16.
//

#ifndef DYNAMICSTEREO_RANDOMFOREST_H
#define DYNAMICSTEREO_RANDOMFOREST_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


namespace dynamic_stereo {

	struct SegmentFeature {
		std::vector<float> feature;
		int id;
	};

	struct FeatureOption {

	};

	using TrainSet = std::vector<std::vector<SegmentFeature> >;

	//remove empty labels in video segments
	void compressSegments(std::vector<cv::Mat>& segments);

	//re-format video segments:
	int regroupSegments(const std::vector<cv::Mat> &segments,
	                    std::vector<std::vector<std::vector<int> > > &pixelGroup,
	                    std::vector<std::vector<int> > &regionSpan);

	void assignSegmentLabel(const std::vector<std::vector<std::vector<int> > >& pixelGroup, const cv::Mat& mask,
	                        std::vector<int>& label);

	void extractFeature(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &segments, const cv::Mat &mask,
	                    const FeatureOption &option, TrainSet &trainSet);

	void visualizeSegmentGroup(const std::vector<cv::Mat> &images, const std::vector<std::vector<int> > &pixelGroup,
	                           const std::vector<int> &regionSpan);

	void visualizeSegmentLabel(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& segments,
	                           const std::vector<int>& label);

}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_RANDOMFOREST_H
