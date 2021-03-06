//
// Created by yanhang on 4/21/16.
//

#ifndef SUBSPACESTAB_TRACKING_H
#define SUBSPACESTAB_TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <glog/logging.h>

namespace substab {

	struct TrackingOption{
		int n_level = 3;
		cv::Size win_size = cv::Size(21,21);
		double quality_level = 0.01;
		double min_distance = 3;

		double max_diff_distance = 1;
		int max_num_corners = 600;
	};

	struct FeatureTracks{
		std::vector<std::vector<cv::Point2f> > tracks;
		std::vector<size_t> offset;
	};

	namespace Tracking {
		void genTrackMatrix(const std::vector<cv::Mat>& images, FeatureTracks& trackMatrix, const int tWindow, const int stride);
		void filterDynamicTracks(FeatureTracks& trackMatrix, const int N);

		void visualizeTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix,
							std::vector<cv::Mat>& output, const int startFrame = 0);
	}
}
#endif //SUBSPACESTAB_TRACKING_H
