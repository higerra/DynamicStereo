//
// Created by yanhang on 10/11/16.
//

#include "region_feature.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    namespace video_segment{
        RegionTransitionPattern::RegionTransitionPattern(const int kFrames, const int s1, const int s2, const float theta,
                                                         const DistanceMetricBase* pixel_distance,
                                                         const TemporalFeatureExtractorBase* spatial_extractor)
                :spatial_extractor_(CHECK_NOTNULL(spatial_extractor)){
            transition_pattern_.reset(new TransitionPattern(kFrames, s1, s2, theta, pixel_distance));
            FeatureBase::dim_ = CHECK_NOTNULL(transition_pattern_.get())->getDim();
            FeatureBase::comparator_.reset(new DistanceHammingAverage());
        }

        void RegionTransitionPattern::ExtractFromPixelFeatures(const cv::_InputArray &pixel_features,
                                                               const std::vector<Region *> &region,
                                                               const cv::OutputArray output) const {
            CHECK(!pixel_features.empty());
            CHECK(!region.empty());
            vector<Mat> pixel_feature_array;
            pixel_features.getMatVector(pixel_feature_array);
            CHECK_EQ(pixel_feature_array.size(), transition_pattern_->GetKFrames());
            const int kPixelFeatureDim = pixel_feature_array[0].cols;
            const int kFrames = pixel_feature_array.size();

            vector<Mat> region_features(kFrames);
            for(auto& m: region_features){
                m.create((int)region.size(), spatial_extractor_->getDim(), pixel_feature_array[0].type());
            }

#pragma omp parallel for
            for (int rid = 0; rid < region.size(); ++rid) {
                const int kPix = CHECK_NOTNULL(region[rid])->pix_id.size();
                for (int v = 0; v < kFrames; ++v) {
                    vector<Mat> region_spatial_features(kPix);
                    int index = 0;
                    for (auto pid: region[rid]->pix_id) {
                        region_spatial_features[index++] = pixel_feature_array[v].row(pid);
                    }
                    Mat tmp;
                    spatial_extractor_->computeFromPixelFeature(region_spatial_features, tmp);
                    tmp.copyTo(region_features[v].row(rid));
                }
            }

            transition_pattern_->computeFromPixelFeature(region_features, output);
        }

    }//namespace video_segment

}//namespace dynamic_stereo