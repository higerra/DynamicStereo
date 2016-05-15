//
// Created by yanhang on 5/15/16.
//

#ifndef DYNAMICSTEREO_TEMPMEANSHIFT_H
#define DYNAMICSTEREO_TEMPMEANSHIFT_H
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "../external/segment_ms/ms.h"

namespace dynamic_stereo{
    void importData(const std::string& path, std::vector<std::vector<float> >& array, const int downsample, const int tWindow);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TEMPMEANSHIFT_H
