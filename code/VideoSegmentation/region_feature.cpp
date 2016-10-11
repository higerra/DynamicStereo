//
// Created by yanhang on 10/11/16.
//

#include "region_feature.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    namespace video_segment{

        RegionTransitionPattern::RegionTransitionPattern(const int kFrames, const int s1, const int s2, const float theta,
                                                         const DistanceMetricBase* pixel_distance)
                :transition_pattern_(new TransitionPattern(kFrames, s1, s2, theta, pixel_distance)){

        }

        void RegionTransitionPattern::ExtractFromPixelFeatures(const cv::_InputArray &pixel_features,
                                                               const std::vector<Region *> &region) const {
            CHECK(!pixel_features.empty());
            vector<Mat> pixel_feature_array;
            pixel_features.getMatVector(pixel_feature_array);


        }

    }//namespace video_segment

}//namespace dynamic_stereo