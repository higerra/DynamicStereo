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


        TransitionAndAppearance::TransitionAndAppearance(const PixelFeatureExtractorBase* pf_,
                                                         const int s1_, const int s2_, const float theta_,
                                                         const double weight_transition, const double weight_appearance)
                : transition_feature_extractor_(new TransitionPattern(pf_, s1_, s2_, theta_)), kBinAppearance_(3){
            sub_weights_.resize(2);
            sub_weights_[0] = weight_appearance;
            sub_weights_[1] = weight_transition;

            sub_comparators_.resize(2);
            sub_comparators_[0].reset(new DistanceL2());
            sub_comparators_[1].reset(CHECK_NOTNULL(transition_feature_extractor_->getDefaultComparator()));
            comparator_.reset(new DistanceCombinedWeighting(vector<size_t>{GetkBinAppearance()}, sub_weights_, sub_comparators_));
        }

        void TransitionAndAppearance::computeFromPixelFeature(const cv::_InputArray &pixelFeatures,
                                                              const cv::_OutputArray &feats) const {
            CHECK(!pixelFeatures.empty());
            Mat transition_features;
            transition_feature_extractor_->computeFromPixelFeature(pixelFeatures, transition_features);
            
        }


    }//video_segment
}//namespace dynamic_stereo

