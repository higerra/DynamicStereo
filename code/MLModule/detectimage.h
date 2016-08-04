//
// Created by yanhang on 8/4/16.
//

#ifndef DYNAMICSTEREO_DETECTIMAGE_H
#define DYNAMICSTEREO_DETECTIMAGE_H

#include "regiondescriptor.h"

namespace dynamic_stereo{
    namespace ML {
        void detectImage(const std::vector<cv::Mat> &images, const std::vector<cv::Mat>& segmentation,
                         cv::Ptr<cv::ml::StatModel> classifier, cv::Mat &output);
    }//namespace ML

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DETECTIMAGE_H
