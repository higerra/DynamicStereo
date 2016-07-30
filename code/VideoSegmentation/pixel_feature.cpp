//
// Created by yanhang on 7/29/16.
//

#include "pixel_feature.h"

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



    ////////////////////////////////////////////////////////////
    //implementation of temporal features
    void TransitionPattern::extractPixel(const VideoMat& input, const int x, const int y, std::vector<FeatureType>& feat) const{
        CHECK(!input.empty());
        feat.clear();
        feat.reserve(2 * input.size());

        vector<PixelType> pix1(3), pix2(3);
        for(auto v=0; v<input.size() - stride1(); ++v){
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
}//namespace dynamic_stereo

