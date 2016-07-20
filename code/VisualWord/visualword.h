//
// Created by yanhang on 7/19/16.
//

#ifndef DYNAMICSTEREO_VISUALWORD_H
#define DYNAMICSTEREO_VISUALWORD_H
#include "../common/HoG3D.h"
#include "../common/regiondescriptor.h"

#include <fstream>

namespace dynamic_stereo {

    struct VisualWordOption{
        VisualWordOption(): M(4), N(4), kSubBlock(3), sigma_s(24), sigma_r(24){}
        int M;
        int N;
        int kSubBlock;
        int sigma_s;
        int sigma_r;
    };

    void sampleKeyPoints(const std::vector<cv::Mat>& input, std::vector<cv::KeyPoint>& keypoints, const VisualWordOption& option);

    void writeTrainData(const std::string& path, const cv::Ptr<cv::ml::TrainData> traindata);

    void writeCodebook(const std::string& path, const cv::Mat& codebook);
    bool loadCodebook(const std::string& path, cv::Mat& codebook);

    double testClassifier(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::StatModel> classifier);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_VISUALWORD_H
