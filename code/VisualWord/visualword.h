//
// Created by yanhang on 7/19/16.
//

#ifndef DYNAMICSTEREO_VISUALWORD_H
#define DYNAMICSTEREO_VISUALWORD_H
#include "../MLModule/CVdescriptor.h"
#include "../MLModule/regiondescriptor.h"
#include "../VideoSegmentation/videosegmentation.h"

#include <fstream>

namespace dynamic_stereo {

    namespace VisualWord {
        enum PixelDescriptor {
            HOG3D,
            COLOR3D
        };

        enum ClassifierType {
            RANDOM_FOREST,
            BOOSTED_TREE,
            SVM
        };

        struct VisualWordOption {
            VisualWordOption() : M(4), N(4), kSubBlock(3), sigma_s(12), sigma_r(24), pixDesc(HOG3D),
                                 classifierType(RANDOM_FOREST) {}

            int M;
            int N;
            int kSubBlock;
            int sigma_s;
            int sigma_r;
            PixelDescriptor pixDesc;
            ClassifierType classifierType;
        };

        void sampleKeyPoints(const std::vector<cv::Mat> &input, std::vector<cv::KeyPoint> &keypoints, const int sigma_s,
                             const int sigma_r);

        void detectVideo(const std::vector<cv::Mat> &images,
                         cv::Ptr<cv::ml::StatModel> classifier, const cv::Mat &codebook,
                         const std::vector<float> &levelList, cv::Mat &output, const VisualWordOption &option);

        void extractSegmentFeature(const std::vector<cv::Mat> &images, const std::vector<ML::PixelGroup> &pixelGroups,
                                   std::vector<std::vector<float> > &feats);

        double testClassifier(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::StatModel> classifier);
    }//namespace VisualWord
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_VISUALWORD_H
