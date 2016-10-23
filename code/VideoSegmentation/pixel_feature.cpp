//
// Created by yanhang on 7/29/16.
//

#include "pixel_feature.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    namespace video_segment {
        /////////////////////////////////////////////////////////////
        //implementation of image features
        void PixelValue::extractAll(const cv::InputArray input, cv::OutputArray output) const {
            CHECK(!input.empty());
            Mat inputMat = input.getMat();
            if(inputMat.depth() == CV_8U) {
                output.create(inputMat.cols * inputMat.rows, inputMat.channels(), CV_8UC1);
            }else if(inputMat.depth() == CV_32F){
                output.create(inputMat.cols * inputMat.rows, inputMat.channels(), CV_32FC1);
            }else{
                CHECK(true) << "Only support CV_8UC and CV_32FC";
            }
            Mat outputMat = output.getMat();
            inputMat.reshape(1, inputMat.cols * inputMat.rows).copyTo(outputMat);
            CHECK_EQ(outputMat.cols, inputMat.channels());
        }

        BRIEFWrapper::BRIEFWrapper() {
            cvBrief = cv::xfeatures2d::BriefDescriptorExtractor::create();
            comparator_.reset(new DistanceHammingAverage());
            FeatureBase::dim_ = cvBrief->descriptorSize();
        }

        void BRIEFWrapper::extractAll(const cv::InputArray input, cv::OutputArray output) const {
            CHECK(!input.empty());
            CHECK(cvBrief.get());
            Mat inputMat = input.getMat();
            const int width = inputMat.cols;
            const int height = inputMat.rows;
            vector<cv::KeyPoint> keypoints((size_t) width * height);
            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    keypoints[y * width + x].pt = cv::Point2f(x, y);
                    //use octave to record pixel id
                    keypoints[y * width + x].octave = y * width + x;
                }
            }
            Mat initRes;
            cvBrief->compute(input, keypoints, initRes);
            //Notice: the BRIEF implementation in OpenCV removes border of the image. Here we add zero padding to the result
            //feature image
            output.create(inputMat.rows * inputMat.cols, initRes.cols, CV_8UC1);
            output.setTo(Scalar::all(0));
            Mat outputMat = output.getMat();
            for (auto i = 0; i < keypoints.size(); ++i) {
                int idx = keypoints[i].octave;
                initRes.row(i).copyTo(outputMat.row(idx));
            }
        }

        ////////////////////////////////////////////////////////////
        //implementation of temporal features

        void TemporalAverage::computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                      const cv::OutputArray feats) const {
            CHECK(!pixelFeatures.empty());

            vector<Mat> feature_arrays;
            pixelFeatures.getMatVector(feature_arrays);

            Mat raw_average(feature_arrays[0].size(), CV_32FC1, cv::Scalar::all(0));
            for(const auto& m: feature_arrays){
                Mat m_float;
                m.convertTo(m_float, CV_32F);
                raw_average += m_float;
            }
            raw_average /= (float)feature_arrays.size();

            feats.create(feature_arrays[0].size(), feature_arrays[0].type());
            Mat feat_mat = feats.getMat();
            raw_average.convertTo(feat_mat, feature_arrays[0].type());
        }


        ColorHistogram::ColorHistogram(const ColorSpace cspace, const vector<int>& kBin,
                                       const int width, const int height, const int R)
                : cspace_(cspace), kBin_(kBin), width_(width), height_(height), R_(R){
            CHECK_EQ(kBin.size(), 3);
            bin_unit_.resize(kBin.size());
            chn_offset_.resize(kBin.size(), 0.0);
            if(cspace == ColorSpace::BGR){
                bin_unit_[0] = 256.0 / (float)kBin[0];
                bin_unit_[1] = 256.0 / (float)kBin[1];
                bin_unit_[2] = 256.0 / (float)kBin[2];
            }else if(cspace == ColorSpace::HSV){
                bin_unit_[0] = 361.0 / (float)kBin[0];
                bin_unit_[1] = 1.01 / (float)kBin[1];
                bin_unit_[2] = 1.01 / (float)kBin[2];
            }else if(cspace == ColorSpace::LAB){
                bin_unit_[0] = 101.0 / (float)kBin[0];
                bin_unit_[1] = 255.0 / (float)kBin[1];
                bin_unit_[2] = 255.0 / (float)kBin[2];
                chn_offset_[1] = -127;
                chn_offset_[2] = -127;
            }

            FeatureBase::dim_ = std::accumulate(kBin.begin(), kBin.end(), 0);
            FeatureBase::comparator_.reset(new DistanceChi2());
            //FeatureBase::comparator_.reset(new DistanceCorrelation());
        }

        void ColorHistogram::computeFromPixelFeature(const cv::_InputArray &pixelFeatures,
                                                     const cv::_OutputArray &feats) const {
            CHECK(!pixelFeatures.empty());
            vector<Mat> pixel_feature_array;
            pixelFeatures.getMatVector(pixel_feature_array);

            const int kPix = pixel_feature_array[0].rows;
            const int K = pixel_feature_array[0].cols;
            CHECK_EQ(K, bin_unit_.size());
            const int N = pixel_feature_array.size();

            vector<Mat> pixel_feature_reshaped(pixel_feature_array.size());
            for(auto v=0; v<N; ++v){
                pixel_feature_reshaped[v] = pixel_feature_array[v].reshape(K, height_);
                CHECK_EQ(pixel_feature_reshaped[v].cols, width_);
            }

            vector<vector<float> > raw_hist((size_t)kPix);
            for(auto& h: raw_hist){
                h.resize((size_t)getDim());
            }

            vector<int> dim_offset(bin_unit_.size(), 0);
            for(auto i=1; i<kBin_.size(); ++i){
                dim_offset[i] = dim_offset[i-1] + kBin_[i-1];
            }

            for(auto v=0; v<N; ++v) {
                Mat color_mat, float_mat;
                pixel_feature_reshaped[v].convertTo(float_mat, CV_32F);
                float_mat /= 255.0;
                if(cspace_ == BGR){
                    color_mat = float_mat * 255.0;
                }else if(cspace_ == HSV) {
                    cvtColor(float_mat, color_mat, CV_BGR2HSV);
                }else if(cspace_ == LAB){
                    cvtColor(float_mat, color_mat, CV_BGR2Lab);
                }

                for (int y = 0; y < height_; ++y) {
                    for (int x = 0; x < width_; ++x) {
                        for (int dx = -R_; dx <= R_; ++dx) {
                            for (int dy = -R_; dy <= R_; ++dy) {
                                int cur_x = x + dx, cur_y = y + dy;
                                if (cur_x >= 0 && cur_x < width_ && cur_y >= 0 && cur_y < height_) {
                                    for (auto c = 0; c < K; ++c) {
                                        int bid = (color_mat.at<Vec3f>(cur_y, cur_x)[c] - chn_offset_[c])/ bin_unit_[c] +
                                                  dim_offset[c];
                                        CHECK_GE(bid, 0);
                                        CHECK_LT(bid, getDim());
                                        raw_hist[y * width_ + x][bid] += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //normalize each histogram s.t. the sum equals 255
            for(auto& h: raw_hist){
                float sum = std::accumulate(h.begin(), h.end(), 0.0);
                for(auto& v: h){
                    v = v / sum * 255;
                }
            }

            feats.create(kPix, getDim(), CV_8UC1);
            Mat feat_mat = feats.getMat();
            for(auto pid=0; pid < feat_mat.rows; ++pid){
                for(auto c=0; c<feat_mat.cols; ++c){
                    feat_mat.at<uchar>(pid, c) = (uchar)raw_hist[pid][c];
                }
            }
        }

        int TransitionPattern::getKChannel(const int kFrames) const {
            int kChannel = 0;
            if(stride1() > 0) {
                for (auto v = 0; v < kFrames - stride1(); v += stride1()) {
                    kChannel++;
                }
            }
            if(stride2() > 0) {
                for (auto v = 0; v < kFrames / 2; v += stride2()) {
                    kChannel++;
                }
            }
            CHECK_GT(kChannel, 0) << "Either stride 1 or stride 2 must be > 0";
            kChannel = std::ceil((float) kChannel / (float) binPerBlock_);
            return kChannel;
        }

        void TransitionPattern::computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                        cv::OutputArray feats) const {
            CHECK(!pixelFeatures.empty());

            vector<Mat> pixelFeatureArray;
            pixelFeatures.getMatVector(pixelFeatureArray);

            const int kPix = pixelFeatureArray[0].rows;
            const int K = pixelFeatureArray[0].cols;
            const int N = (int)pixelFeatureArray.size();

            CHECK_EQ(N, TransitionFeature::kFrames_);

            const int kChannel = getDim();

            feats.create(kPix, kChannel, CV_8UC1);
            Mat featMat = feats.getMat();
            featMat.setTo(Scalar::all(0));

            int featIndex = 0;

            if(stride1() > 0) {
                for (auto v = 0; v < N - stride1(); v += stride1()) {
                    const int blockId = featIndex / binPerBlock_;
                    const int cellId = featIndex % binPerBlock_;
                    for (auto i = 0; i < kPix; ++i) {
                        double d = pixel_distance_->evaluate(pixelFeatureArray[v].row(i),
                                                            pixelFeatureArray[v + stride1()].row(i));
                        if (d >= theta())
                            featMat.at<uchar>(i, blockId) |= or_table_[cellId];
                    }
                    featIndex++;
                }
            }

            if(stride2() > 0) {
                for (auto v = 0; v < N / 2; v += stride2()) {
                    const int blockId = featIndex / binPerBlock_;
                    const int cellId = featIndex % binPerBlock_;
                    for (auto i = 0; i < kPix; ++i) {
                        float d = pixel_distance_->evaluate(pixelFeatureArray[v].row(i),
                                                           pixelFeatureArray[v + N / 2].row(i));
                        if (d >= theta())
                            featMat.at<uchar>(i, blockId) |= or_table_[cellId];
                    }
                    featIndex++;
                }
            }
        }

        void TransitionPattern::printFeature(const cv::InputArray input) const{
            const Mat feat = input.getMat();
            const uchar *pData = feat.data;
            for (auto i = 0; i < feat.rows * feat.cols; ++i) {
                std::bitset<8> bitset1(static_cast<char>(pData[i]));
                std::cout << bitset1;
            }
            std::cout << std::endl;
        }


        CombinedTemporalFeature::CombinedTemporalFeature(const std::vector<std::shared_ptr<TemporalFeatureExtractorBase> > extractors,
                                                         const std::vector<double>& weights,
                                                         const std::vector<std::shared_ptr<DistanceMetricBase> >* sub_comparators)
                :temporal_extractors_(extractors),  weights_(weights) {
            CHECK_EQ(temporal_extractors_.size(), weights_.size());

            FeatureBase::dim_ = 0;
            for(auto i=0; i<temporal_extractors_.size(); ++i){
                FeatureBase::dim_ += CHECK_NOTNULL(temporal_extractors_[i].get())->getDim();
            }
            sub_comparators_.resize(temporal_extractors_.size());
            if(sub_comparators != nullptr){
                CHECK_EQ(sub_comparators->size(), temporal_extractors_.size());
                for(auto i=0; i<sub_comparators->size(); ++i){
                    sub_comparators_[i] = (*sub_comparators)[i];
                }
            }else{
                for(auto i=0; i<sub_comparators_.size(); ++i){
                    sub_comparators_[i] = temporal_extractors_[i]->getDefaultComparatorSmartPointer();
                    CHECK(sub_comparators_[i].get() != nullptr) << "Extractor " << i << " does not have a valid comparator";
                }
            }

            offset_.resize(temporal_extractors_.size(), 0);
            for(auto i=1; i<temporal_extractors_.size(); ++i){
                offset_[i] = temporal_extractors_[i-1]->getDim() + offset_[i-1];
            }

            vector<size_t> splits(weights_.size() - 1);
            for(auto i=0; i<splits.size(); ++i){
                splits[i] = static_cast<size_t>(offset_[i+1]);
            }
            comparator_.reset(new DistanceCombinedWeighting(splits, weights_, sub_comparators_));
        }

        void CombinedTemporalFeature::computeFromPixelFeatures(const std::vector<std::vector<cv::Mat> >& pixel_features,
                                                               const cv::OutputArray feats) const {
            CHECK(!pixel_features.empty());
            CHECK_EQ(pixel_features.size(), temporal_extractors_.size());

            int kPix = pixel_features[0][0].rows;
            feats.create(kPix, getDim(), CV_8UC1);
            Mat feats_mat = feats.getMat();

            for(auto comid = 0; comid < temporal_extractors_.size(); ++comid){
                Mat feature_component;
                temporal_extractors_[comid]->computeFromPixelFeature(pixel_features[comid], feature_component);
                CHECK_EQ(feature_component.type(), feats_mat.type());
                feature_component.copyTo(feats_mat.colRange(offset_[comid], offset_[comid] + temporal_extractors_[comid]->getDim()));
            }
        }

        void CombinedTemporalFeature::printFeature(const cv::InputArray input) const{
            const Mat feat = input.getMat();
            CHECK_EQ(feat.cols, getDim());
            for(auto rid=0; rid < feat.rows; ++rid){
                for(auto comid=0; comid < temporal_extractors_.size(); ++comid){
                    temporal_extractors_[comid]->printFeature(
                            feat.row(rid).colRange(offset_[comid], offset_[comid]+temporal_extractors_[comid]->getDim()));
                }
            }
        }

    }//video_segment
}//namespace dynamic_stereo

