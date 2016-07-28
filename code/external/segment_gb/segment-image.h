/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include "segment-graph.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <glog/logging.h>
#include <memory>
#include <random>

namespace segment_gb {

	class TemporalComparator{
	public:
		virtual float compare(const std::vector<cv::Mat>& input, const int x1, const int y1, const int x2, const int y2) const = 0;
	};

	class TransitionPattern: public TemporalComparator{
	public:
		TransitionPattern(const int param1_, const int param2_, const float param3_): stride1(param1_), stride2(param2_), theta(param3_){}
		virtual float compare(const std::vector<cv::Mat>& input, const int x1, const int y1, const int x2, const int y2) const;
	private:

		const int stride1;
		const int stride2;
		const float theta;
	};

	class TransitionCounter: public TemporalComparator{
	public:
		TransitionCounter(const int param1_, const int param2_, const float param3_): stride1(param1_), stride2(param2_), theta(param3_){}
		virtual float compare(const std::vector<cv::Mat>& input, const int x1, const int y1, const int x2, const int y2) const;
	private:
		const int stride1;
		const int stride2;
		const float theta;
	};

	//input: input image
	//output: Mat with CV_32S type. Pixel values correspond to label Id
	//seg: grouped pixels. seg[i][j], j'th pixel in i'th segment
	int segment_image(const cv::Mat& input, cv::Mat& output, std::vector<std::vector<int> >& seg,
	                   const int smoothSize, float c, int min_size);

	void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output);

	int segment_video(const std::vector<cv::Mat>& input, cv::Mat& output,
	                  const int smoothSize, const float c, const float theta, const int min_size);
	cv::Mat visualizeSegmentation(const cv::Mat& input);
}
#endif
