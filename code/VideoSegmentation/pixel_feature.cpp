//
// Created by yanhang on 7/29/16.
//

#include "pixel_feature.h"
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    /////////////////////////////////////////////////////////////
    //implementation of image features
    void PixelValue::extractPixel(const cv::Mat &input, const int x, const int y, std::vector<float> &feat) const {
        if(feat.size() != 3)
            feat.resize(3);
        Vec3f pix;
        if (input.type() == CV_8UC3) {
            pix = (Vec3f) input.at<Vec3b>(y, x);
        } else if (input.type() == CV_32FC3)
            pix = input.at<Vec3f>(y, x);
        else
            CHECK(true) << "Image must be either CV_8UC3 or CV_32FC3";
        feat[0] = pix[0];
        feat[1] = pix[1];
        feat[2] = pix[2];
    }


    void PixelValue::extractImage(const cv::Mat &input, cv::OutputArray &output) const {
        output.create(input.cols * input.rows, 3, CV_32FC1);
        Mat outputMat = output.getMat();

        for(auto y=0; y<input.rows; ++y){
            for(auto x=0; x<input.cols; ++x){
                Vec3f pix;
                if (input.type() == CV_8UC3) {
                    pix = (Vec3f) input.at<Vec3b>(y, x);
                } else if (input.type() == CV_32FC3)
                    pix = input.at<Vec3f>(y, x);
                else
                    CHECK(true) << "Image must be either CV_8UC3 or CV_32FC3";
                for(auto j=0; j<3; ++j)
                    outputMat.at<float>(y*input.cols+x, j) = pix[j];
            }
        }
    }

    BRIEFWrapper::BRIEFWrapper() {
        cvBrief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }

    void BRIEFWrapper::extractImage(const cv::Mat &input, cv::OutputArray& output) const {
        CHECK(input.data);
        CHECK(cvBrief.get());
        const int width = input.cols;
        const int height = input.rows;
        vector<cv::KeyPoint> keypoints((size_t)width * height);
        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){
                keypoints[y*width+x].pt = cv::Point2f(x,y);
            }
        }
        cvBrief->compute(input, keypoints, output);
    }

    void BRIEFWrapper::extractPixel(const cv::Mat &input, const int x, const int y, std::vector<uchar> &feat) const {
        vector<cv::KeyPoint> kpt(1);
        kpt[0].pt = cv::Point2f(x,y);
        CHECK_NOTNULL(cvBrief.get())->compute(input, kpt, feat);
    }


    ////////////////////////////////////////////////////////////
    //implementation of temporal features
    void TransitionPattern::extractPixel(const VideoMat& input, const int x, const int y, std::vector<FeatureType>& feat) const{
        CHECK(!input.empty());
        feat.clear();
        feat.reserve(2 * input.size());

        vector<PixelType> pix1(3), pix2(3);
        for(auto v=0; v<input.size() - stride1(); v += stride1()){
            pixel_feature->extractPixel(input[v], x, y, pix1);
            pixel_feature->extractPixel(input[v+stride1()], x, y, pix2);
            float d = pixel_distance->evaluate(pix1, pix2);
            if(d >= theta())
                feat.push_back(true);
            else
                feat.push_back(false);
        }

        if(stride2() > 0){
            for(auto v=0; v<input.size() - stride2(); v+=stride1()/2) {
                pixel_feature->extractPixel(input[v], x, y, pix1);
                pixel_feature->extractPixel(input[v+stride2()], x, y, pix2);
                float d = pixel_distance->evaluate(pix1, pix2);
                if (d >= theta())
                    feat.push_back(true);
                else
                    feat.push_back(false);
            }
        }
    }

    void TransitionPattern::computeFromPixelFeature(const VideoMat &pixelFeatures,
                                                    std::vector<std::vector<FeatureType> > &feats) const {
        CHECK(!pixelFeatures.empty());
        const int kPix = pixelFeatures[0].rows;
        const int K = pixelFeatures[0].cols;

        feats.resize((size_t)kPix);

        for(auto& feat: feats)
            feat.reserve(2 * pixelFeatures.size());

        for(auto v=0; v<pixelFeatures.size() - stride1(); v += stride1()){
            const PixelType* pPixel1 = (const PixelType*) pixelFeatures[v].data;
            const PixelType* pPixel2 = (const PixelType*) pixelFeatures[v+stride1()].data;
            for(auto i=0; i<kPix; ++i){
                vector<PixelType> pix1(pPixel1+i*K, pPixel1+(i+1)*K);
                vector<PixelType> pix2(pPixel2+i*K, pPixel2+(i+1)*K);
                FeatureType d = pixel_distance->evaluate(pix1, pix2);
                if(d >= theta())
                    feats[i].push_back(true);
                else
                    feats[i].push_back(false);
            }
        }

        if(stride2() > 0){
            for(auto v=0; v<pixelFeatures.size() - stride2(); v+=stride1()/2) {
                const PixelType* pPixel1 = (const PixelType*) pixelFeatures[v].data;
                const PixelType* pPixel2 = (const PixelType*) pixelFeatures[v+stride2()].data;
                for(auto i=0; i<kPix; ++i){
                    vector<PixelType> pix1(pPixel1+i*K, pPixel1+(i+1)*K);
                    vector<PixelType> pix2(pPixel2+i*K, pPixel2+(i+1)*K);
                    float d = pixel_distance->evaluate(pix1, pix2);
                    if (d >= theta())
                        feats[i].push_back(true);
                    else
                        feats[i].push_back(false);
                }
            }
        }
    }

    void TransitionCounting::extractPixel(const VideoMat &input, const int x, const int y,
                                          std::vector<float> &feat) const {
        CHECK_GE(input.size(), 2);
        feat.resize(2, 0.0f);

	    vector<PixelType> pix1(3), pix2(3);
        float counter1 = 0.0f, counter2 = 0.0f;
        for(auto v=0; v<input.size() - stride1(); v+=stride1()){
            pixel_feature->extractPixel(input[v], x, y, pix1);
            pixel_feature->extractPixel(input[v+stride1()], x, y, pix2);
            float d = pixel_distance->evaluate(pix1, pix2);
            if(d >= theta())
                feat[0] += 1.0f;
            counter1 += 1.0;
        }

        for(auto v=0; v<input.size() - stride2(); v+=stride1()/2) {
            pixel_feature->extractPixel(input[v], x, y, pix1);
            pixel_feature->extractPixel(input[v+stride2()], x, y, pix2);
            float d = pixel_distance->evaluate(pix1, pix2);
            if(d >= theta())
                feat[1] += 1.0f;
            counter2 += 1.0;
        }
        feat[0] /= counter1;
        feat[1] /= counter2;
    }

    void TransitionCounting::computeFromPixelFeature(const VideoMat &pixelFeatures,
                                                     std::vector<std::vector<FeatureType> > &feats) const {

    }
}//namespace dynamic_stereo

