//
// Created by yanhang on 6/7/16.
//

#ifndef DYNAMICSTEREO_CONTOURDEV_H
#define DYNAMICSTEREO_CONTOURDEV_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include "../external/video_segmentation/segment_util/segmentation_util.h"

namespace dynamic_stereo{

    segmentation::SegmentationDesc readSegmentation(const std::string& path);

    void filterSegment(const segmentation::SegmentationDesc& input, std::vector<cv::Mat>& output);


}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_CONTOURDEV_H
