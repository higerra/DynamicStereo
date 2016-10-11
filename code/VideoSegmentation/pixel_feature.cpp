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
        void
        PixelValue::extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const {
            output.create(getDim(), 1, CV_32FC1);
            Mat inputMat = input.getMat();
            Mat outputMat = output.getMat();
            Vec3f pix;
            if (input.type() == CV_8UC3) {
                pix = (Vec3f) inputMat.at<Vec3b>(y, x);
            } else if (input.type() == CV_32FC3)
                pix = inputMat.at<Vec3f>(y, x);
            else
                CHECK(true) << "Image must be either CV_8UC3 or CV_32FC3";
            outputMat.at<float>(0, 0) = pix[0];
            outputMat.at<float>(1, 0) = pix[1];
            outputMat.at<float>(2, 0) = pix[2];
        }


        void PixelValue::extractAll(const cv::InputArray input, cv::OutputArray output) const {
            Mat inputMat = input.getMat();
            output.create(inputMat.cols * inputMat.rows, 3, CV_32FC1);
            Mat outputMat = output.getMat();

            for (auto y = 0; y < inputMat.rows; ++y) {
                for (auto x = 0; x < inputMat.cols; ++x) {
                    Vec3f pix;
                    if (input.type() == CV_8UC3) {
                        pix = (Vec3f) inputMat.at<Vec3b>(y, x);
                    } else if (input.type() == CV_32FC3)
                        pix = inputMat.at<Vec3f>(y, x);
                    else
                        CHECK(true) << "Image must be either CV_8UC3 or CV_32FC3";
                    for (auto j = 0; j < 3; ++j)
                        outputMat.at<float>(y * inputMat.cols + x, j) = pix[j];
                }
            }
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

        void
        BRIEFWrapper::extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const {
            vector<cv::KeyPoint> kpt(1);
            kpt[0].pt = cv::Point2f(x, y);
            CHECK_NOTNULL(cvBrief.get())->compute(input, kpt, feat);
        }

        ////////////////////////////////////////////////////////////
        //implementation of temporal features

        void TemporalAverage::computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                      const cv::OutputArray feats) const {
            CHECK(!pixelFeatures.empty());

            vector<Mat> feature_arrays;
            pixelFeatures.getMatVector(feature_arrays);

            Mat raw_average(feature_arrays[0].size(), CV_32FC1);
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
                : kBin_(kBin), width_(width), height_(height), R_(R){
            CHECK_EQ(kBin.size(), 3);
            bin_unit.resize(kBin.size());
            chn_offset.resize(kBin.size(), 0.0);
            if(cspace == ColorSpace::BGR){
                bin_unit[0] = 256.0 / (float)kBin[0];
                bin_unit[1] = 256.0 / (float)kBin[1];
                bin_unit[2] = 256.0 / (float)kBin[2];
            }else if(cspace == ColorSpace::HSV){
                bin_unit[0] = 361.0 / (float)kBin[0];
                bin_unit[1] = 1.01 / (float)kBin[1];
                bin_unit[2] = 1.01 / (float)kBin[2];
            }else if(cspace == ColorSpace::LAB){
                bin_unit[0] = 101.0 / (float)kBin[0];
                bin_unit[1] = 255.0 / (float)kBin[1];
                bin_unit[2] = 255.0 / (float)kBin[2];
                chn_offset[1] = -127;
                chn_offset[2] = -127;
            }

            dim_ = std::accumulate(kBin.begin(), kBin.end(), 0);
        }

        void ColorHistogram::computeFromPixelFeature(const cv::_InputArray &pixelFeatures,
                                                     const cv::_OutputArray &feats) const {
            CHECK(!pixelFeatures.empty());
            vector<Mat> pixel_feature_array;
            pixelFeatures.getMatVector(pixel_feature_array);

            const int kPix = pixel_feature_array[0].rows;
            CHECK_EQ(width_ * height_, kPix);
            const int K = pixel_feature_array[0].cols;
            CHECK_EQ(K, bin_unit.size());
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

            vector<int> dim_offset(bin_unit.size(), 0);
            for(auto i=1; i<kBin_.size(); ++i){
                dim_offset[i] = dim_offset[i-1] + kBin_[i-1];
            }

            for(auto v=0; v<N; ++v) {
                Mat float_mat;
                pixel_feature_reshaped[v].convertTo(float_mat, CV_32F);
                float_mat /= 255.0;
                Mat hsv_mat;
                cvtColor(float_mat, hsv_mat, CV_BGR2HSV);
                for (auto y = 0; y < height_; ++y) {
                    for (auto x = 0; x < width_; ++x) {
                        for (auto dx = -R_; dx <= R_; ++dx) {
                            for (auto dy = -R_; dy <= R_; ++dy) {
                                int cur_x = x + dx, cur_y = y + dy;
                                if (cur_x >= 0 && cur_x < width_ && cur_y >= 0 && cur_y < height_) {
                                    for (auto c = 0; c < K; ++c) {
                                        int bid = hsv_mat.at<Vec3f>(cur_y, cur_x)[c] / bin_unit[c] +
                                                  dim_offset[c];
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
            kChannel = std::ceil((float) kChannel / (float) binPerCell_);
            return kChannel;
        }

        void TransitionPattern::extractPixel(const cv::InputArray input, const int x, const int y,
                                             cv::OutputArray feat) const {
            CHECK(!input.empty());

            vector<Mat> inputArray;
            input.getMatVector(inputArray);

            //first compute the dimension of the feature vector
            const int kChannel = getKChannel((int) inputArray.size());
            const int N = (int)inputArray.size();

            feat.create(kChannel, 1, CV_8UC1);
            feat.setTo(cv::Scalar::all(0));
            Mat featMat = feat.getMat();

            Mat pix1, pix2;
            int featIndex = 0;
            if(stride1() > 0) {
                for (auto v = 0; v < N - stride1(); v += stride1()) {
                    const int blockId = featIndex / binPerCell_;
                    const int cellId = featIndex % binPerCell_;
                    pixel_feature->extractPixel(inputArray[v], x, y, pix1);
                    pixel_feature->extractPixel(inputArray[v + stride1()], x, y, pix2);
                    float d = pixel_distance->evaluate(pix1, pix2);
                    if (d >= theta())
                        featMat.at<uchar>(blockId, 0) |= or_table_[cellId];
                    featIndex++;
                }
            }

            if(stride2() > 0) {
                for (auto v = 0; v < N / 2; v += stride2()) {
                    const int blockId = featIndex / binPerCell_;
                    const int cellId = featIndex % binPerCell_;
                    pixel_feature->extractPixel(inputArray[v], x, y, pix1);
                    pixel_feature->extractPixel(inputArray[v + N / 2], x, y, pix2);
                    float d = pixel_distance->evaluate(pix1, pix2);
                    if (d >= theta())
                        featMat.at<uchar>(blockId, 0) |= or_table_[cellId];
                    featIndex++;
                }
            }
        }

        void TransitionPattern::computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                        cv::OutputArray feats) const {
            CHECK(!pixelFeatures.empty());

            vector<Mat> pixelFeatureArray;
            pixelFeatures.getMatVector(pixelFeatureArray);

            const int kPix = pixelFeatureArray[0].rows;
            const int K = pixelFeatureArray[0].cols;
            const int N = (int)pixelFeatureArray.size();

            const int kChannel = getKChannel((int) pixelFeatureArray.size());
            feats.create(kPix, kChannel, CV_8UC1);
            Mat featMat = feats.getMat();
            featMat.setTo(Scalar::all(0));

            int featIndex = 0;

            if(stride1() > 0) {
                for (auto v = 0; v < N - stride1(); v += stride1()) {
                    const int blockId = featIndex / binPerCell_;
                    const int cellId = featIndex % binPerCell_;
                    for (auto i = 0; i < kPix; ++i) {
                        double d = pixel_distance->evaluate(pixelFeatureArray[v].row(i),
                                                            pixelFeatureArray[v + stride1()].row(i));
                        if (d >= theta())
                            featMat.at<uchar>(i, blockId) |= or_table_[cellId];
                    }
                    featIndex++;
                }
            }

            if(stride2() > 0) {
                for (auto v = 0; v < N / 2; v += stride2()) {
                    const int blockId = featIndex / binPerCell_;
                    const int cellId = featIndex % binPerCell_;
                    for (auto i = 0; i < kPix; ++i) {
                        float d = pixel_distance->evaluate(pixelFeatureArray[v].row(i),
                                                           pixelFeatureArray[v + N / 2].row(i));
                        if (d >= theta())
                            featMat.at<uchar>(i, blockId) |= or_table_[cellId];
                    }
                    featIndex++;
                }
            }
        }

        void TransitionPattern::printFeature(const cv::InputArray input) {
            const Mat feat = input.getMat();
            const uchar *pData = feat.data;
            for (auto i = 0; i < feat.rows * feat.cols; ++i) {
                std::bitset<8> bitset1(static_cast<char>(pData[i]));
                std::cout << bitset1;
            }
            std::cout << std::endl;
        }

        void TransitionCounting::extractPixel(const cv::InputArray input, const int x, const int y,
                                              cv::OutputArray feat) const {
//        CHECK_GE(input.size(), 2);
//        feat.resize(2, 0.0f);
//
//	    vector<PixelType> pix1(3), pix2(3);
//        float counter1 = 0.0f, counter2 = 0.0f;
//        for(auto v=0; v<input.size() - stride1(); v+=stride1()){
//            pixel_feature->extractPixel(input[v], x, y, pix1);
//            pixel_feature->extractPixel(input[v+stride1()], x, y, pix2);
//            float d = pixel_distance->evaluate(pix1, pix2);
//            if(d >= theta())
//                feat[0] += 1.0f;
//            counter1 += 1.0;
//        }
//
//        for(auto v=0; v<input.size() - stride2(); v+=stride1()/2) {
//            pixel_feature->extractPixel(input[v], x, y, pix1);
//            pixel_feature->extractPixel(input[v+stride2()], x, y, pix2);
//            float d = pixel_distance->evaluate(pix1, pix2);
//            if(d >= theta())
//                feat[1] += 1.0f;
//            counter2 += 1.0;
//        }
//        feat[0] /= counter1;
//        feat[1] /= counter2;
        }

        void TransitionCounting::computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                         cv::OutputArray feats) const {

        }


        CombinedTemporalFeature::CombinedTemporalFeature(const std::vector<std::shared_ptr<TemporalFeatureExtractorBase> > extractors,
                                                         const std::vector<double>& weights,
                                                         const std::vector<std::shared_ptr<DistanceMetricBase> >* sub_comparators)
                :temporal_extractors_(extractors),  weights_(weights) {
            CHECK_EQ(temporal_extractors_.size(), weights_.size());

            dim_ = 0;
            for(auto i=0; i<temporal_extractors_.size(); ++i){
                dim_ += CHECK_NOTNULL(temporal_extractors_[i].get())->getDim();
            }
            sub_comparators_.resize(temporal_extractors_.size());
            if(sub_comparators != nullptr){
                CHECK_EQ(sub_comparators->size(), temporal_extractors_.size());
                for(auto i=0; i<sub_comparators->size(); ++i){
                    sub_comparators_[i] = (*sub_comparators)[i];
                }
            }else{
                for(auto i=0; i<sub_comparators->size(); ++i){
                    sub_comparators_[i] = temporal_extractors_[i]->getDefaultComparatorSmartPointer();
                }
            }

            offset_.resize(temporal_extractors_.size(), 0);
            for(auto i=1; i<temporal_extractors_.size(); ++i){
                offset_[i] = temporal_extractors_[i-1]->getDim() + offset_[i-1];
            }
        }

        void CombinedTemporalFeature::computeFromPixelFeatures(const std::vector<Mat>& pixel_features,
                                                               const cv::OutputArray feats) const {
            CHECK_EQ(pixel_features.size(), temporal_extractors_.size());
            int kPix = pixel_features[0].rows;
            feats.create(kPix, getDim(), CV_8UC1);
            
        }

        void CombinedTemporalFeature::printFeature(const cv::InputArray input){
            const Mat feat = input.getMat();
            const uchar *pData = feat.data;
            for(auto i=0; i<GetkBinAppearance(); ++i){
                std::cout << (int)pData[i]<< ' ';
            }
            for (auto i = GetkBinAppearance(); i < feat.rows * feat.cols; ++i) {
                std::bitset<8> bitset1(static_cast<char>(pData[i]));
                std::cout << bitset1;
            }
            std::cout << std::endl;
        }

    }//video_segment
}//namespace dynamic_stereo

