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

	//input: input image
	//output: Mat with CV_32S type. Pixel values correspond to label Id
	//seg: grouped pixels. seg[i][j], j'th pixel in i'th segment
	int segment_image(const cv::Mat& input, cv::Mat& output, std::vector<std::vector<int> >& seg,
	                   const int smoothSize, float c, int min_size);

	int compressSegment(cv::Mat& segment);
	cv::Mat visualizeSegmentation(const cv::Mat& input);
}
#endif
