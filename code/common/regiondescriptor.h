//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_REGIONDESCRIPTOR_H
#define DYNAMICSTEREO_REGIONDESCRIPTOR_H

#include "descriptor.h"

namespace dynamic_stereo {
	namespace Feature {
		struct SegmentFeature {
			std::vector<float> feature;
			int id;
		};

		struct FeatureOption {

		};

		using TrainSet = std::vector<std::vector<SegmentFeature> >;

		//remove empty labels in video segments
		int compressSegments(std::vector<cv::Mat>& segments);

		//re-format video segments:
		int regroupSegments(const std::vector<cv::Mat> &segments,
		                    std::vector<std::vector<std::vector<int> > > &pixelGroup,
		                    std::vector<std::vector<int> > &regionSpan);

		void assignSegmentLabel(const std::vector<std::vector<std::vector<int> > >& pixelGroup, const cv::Mat& mask,
		                        std::vector<int>& label);

		void computeHoG(const std::vector<cv::Mat>& gradient, const std::vector<std::vector<int> >& pixelIds,
		                std::vector<float>& hog, const int kBin);

		void extractFeature(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &segments, const cv::Mat &mask,
		                    const FeatureOption &option, TrainSet &trainSet);

		void visualizeSegmentGroup(const std::vector<cv::Mat> &images, const std::vector<std::vector<int> > &pixelGroup,
		                           const std::vector<int> &regionSpan);

		void visualizeSegmentLabel(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& segments,
		                           const std::vector<int>& label);

	}//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_REGIONDESCRIPTOR_H
