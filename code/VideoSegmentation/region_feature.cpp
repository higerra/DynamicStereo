//
// Created by yanhang on 10/11/16.
//

#include "region_feature.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    namespace video_segment{

        RegionColorHist::RegionColorHist(const ColorHistogram::ColorSpace cspace, const std::vector<int>& kBin,
                                         const int width, const int height)
                :cspace_(cspace) , width_(width), height_(height){
            CHECK_EQ(kBin.size(), 3);
            bin_unit_.resize(kBin.size());
            chn_offset_.resize(kBin.size(), 0.0);
            if(cspace == ColorHistogram::ColorSpace::BGR){
                bin_unit_[0] = 256.0 / (float)kBin[0];
                bin_unit_[1] = 256.0 / (float)kBin[1];
                bin_unit_[2] = 256.0 / (float)kBin[2];
            }else if(cspace == ColorHistogram::ColorSpace::HSV){
                bin_unit_[0] = 361.0 / (float)kBin[0];
                bin_unit_[1] = 1.01 / (float)kBin[1];
                bin_unit_[2] = 1.01 / (float)kBin[2];
            }else if(cspace == ColorHistogram::LAB){
                bin_unit_[0] = 101.0 / (float)kBin[0];
                bin_unit_[1] = 255.0 / (float)kBin[1];
                bin_unit_[2] = 255.0 / (float)kBin[2];
                chn_offset_[1] = -127;
                chn_offset_[2] = -127;
            }
            FeatureBase::dim_ = std::accumulate(kBin.begin(), kBin.end(), 0);
            FeatureBase::comparator_.reset(new DistanceChi2());
        }

        void RegionColorHist::ExtractFromPixelFeatures(const cv::_InputArray &pixel_features,
                                                       const std::vector<Region *> &region,
                                                       const cv::_OutputArray &output) const {
            CHECK(!pixel_features.empty());
            CHECK(!region.empty());
            vector<Mat> pixel_feature_array;
            pixel_features.getMatVector(pixel_feature_array);

            const int K = pixel_feature_array[0].cols;
            CHECK_EQ(K, kBin_.size());

            vector<vector<float> > raw_hist(region.size());
            for(auto& h: raw_hist){
                h.resize(getDim(), 0.0);
            }

            vector<int> dim_offset(bin_unit_.size(), 0);
            for(auto i=1; i<kBin_.size(); ++i){
                dim_offset[i] = dim_offset[i-1] + kBin_[i-1];
            }

            for(auto v=0; v<pixel_feature_array.size(); ++v){
                Mat frame_reshaped = pixel_feature_array[v].reshape(K, height_);
                CHECK_EQ(frame_reshaped.cols, width_);
                Mat color_mat, float_mat;
                frame_reshaped.convertTo(float_mat, CV_32F);
                float_mat /= 255.0;
                if(cspace_ == ColorHistogram::BGR){
                    color_mat = float_mat * 255.0;
                }else if(cspace_ == ColorHistogram::HSV) {
                    cvtColor(float_mat, color_mat, CV_BGR2HSV);
                }else if(cspace_ == ColorHistogram::LAB){
                    cvtColor(float_mat, color_mat, CV_BGR2Lab);
                }

                for(auto rid=0; rid < region.size(); ++rid){
                    CHECK(region[rid]);
                    for(auto pid: region[rid]->pix_id){
                        const float* pix = (float*) color_mat.data + pid * K;
                        for (auto c = 0; c < K; ++c) {
                            int bid = (pix[c] - chn_offset_[c])/ bin_unit_[c] +
                                      dim_offset[c];
                            CHECK_GE(bid, 0);
                            CHECK_LT(bid, getDim());
                            raw_hist[rid][bid] += 1.0;
                        }
                    }
                }

            }

            output.create((int)region.size(), getDim(), CV_8UC1);
            Mat output_mat = output.getMat();
            output_mat.setTo(cv::Scalar::all(0));

            for(auto rid=0; rid < region.size(); ++rid){
                float sum = std::accumulate(raw_hist[rid].begin(), raw_hist[rid].end(), 0.0);
                if(sum > numeric_limits<float>::epsilon()){
                    for(auto& h: raw_hist[rid]){
                        h /= sum * 255;
                    }
                }
                for(auto c=0; c<getDim(); ++c){
                    output_mat.at<uchar>(rid, c) = (uchar)raw_hist[rid][c];
                }
            }

        }




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