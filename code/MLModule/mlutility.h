//
// Created by yanhang on 8/3/16.
//

#ifndef DYNAMICSTEREO_MLUTILITY_H
#define DYNAMICSTEREO_MLUTILITY_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "type_def.h"

namespace dynamic_stereo {
    namespace MLUtility {
        cv::Ptr<cv::ml::TrainData> convertTrainData(const Feature::TrainSet &trainset);

        void writeTrainData(const std::string &path, const cv::Ptr<cv::ml::TrainData> traindata);

        void normalizel2(std::vector<float> &array);

        void normalizeSum(std::vector<float> &array);

        inline void normalizel1(std::vector<float> &array) {
            normalizeSum(array);
        }
    };
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_MLUTILITY_H
