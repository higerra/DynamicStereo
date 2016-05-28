//
// Created by yanhang on 5/28/16.
//

#include "gtest/gtest.h"
#include "descriptor.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

TEST(FeatureModule, ColorSpace){
    Feature::ColorSpace c1(Feature::ColorSpace::RGB);
    Feature::ColorSpace c2(Feature::ColorSpace::LUV);
    EXPECT_EQ(c1.channel, 3);
    EXPECT_EQ(c2.channel, 3);
    for(auto c=0; c<3; ++c) {
        EXPECT_NEAR(c1.offset[c], 0, FLT_EPSILON);
        EXPECT_NEAR(c1.range[c], 256, FLT_EPSILON);
    }
    EXPECT_NEAR(c2.offset[0], 0, FLT_EPSILON);
    EXPECT_NEAR(c2.offset[1], -134, FLT_EPSILON);
    EXPECT_NEAR(c2.offset[2], -140, FLT_EPSILON);
    EXPECT_NEAR(c2.range[0], 100, FLT_EPSILON);
    EXPECT_NEAR(c2.range[1], 354, FLT_EPSILON);
    EXPECT_NEAR(c2.range[2], 262, FLT_EPSILON);
}

TEST(FeatureModule, descriptor_hist){
    const int tW = 100, tH = 100;
    Mat ranImgRGB(tH,tW,CV_8UC3);
    cv::randu(ranImgRGB, Scalar::all(0), Scalar::all(255));
    Mat ranImgRGBFloat;
    ranImgRGB.convertTo(ranImgRGBFloat, CV_32FC3);
    ranImgRGBFloat /= 255;
    Mat ranImgLUV;
    cvtColor(ranImgRGBFloat, ranImgLUV, CV_RGB2Luv);
    const uchar* pRGB = ranImgRGB.data;
    for(auto i=0; i<tW * tH; ++i){
        CHECK_GE((int)pRGB[i], 0);
        CHECK_LT((int)pRGB[i], 256);
    }

    const float* pLuv = (float*)ranImgLUV.data;
    vector<float> arrayRGB(100*100*3,0), arrayLuv(100*100*3,0);
    for(auto i=0; i<tH*tW; ++i){
        for(auto c=0; c<3; ++c){
            arrayRGB[i*3+c] = (float)pRGB[i*3+c];
            arrayLuv[i*3+c] = pLuv[i*3+c];
            CHECK_GE(arrayLuv[i*3+c], -200) << i%tW << ' ' << i/tW << ' ' << c;
            CHECK_LT(arrayLuv[i*3+c], 200) << i%tW << ' ' <<i/tW << ' ' << c;
        }
    }
    Feature::ColorSpace cRGB(Feature::ColorSpace::RGB);
    Feature::ColorSpace cLUV(Feature::ColorSpace::LUV);
    vector<int> kBin{10,10,10};
    Feature::RGBHist rgbHist(10);
    Feature::ColorHist colorHist(cRGB, kBin);
    Feature::ColorHist luvHist(cLUV, kBin);

    vector<float> featRGB1, featRGB2, featLUV;
    rgbHist.constructFeature(arrayRGB, featRGB1);
    colorHist.constructFeature(arrayRGB, featRGB2);
    luvHist.constructFeature(arrayLuv, featLUV);
    float diff = 0.0;
    EXPECT_EQ(featRGB1.size(), featRGB2.size());
    EXPECT_EQ(featRGB1.size(), featLUV.size());
    for(auto i=0; i<featRGB1.size(); ++i)
        diff += featRGB1[i] - featRGB2[i];
    diff = std::sqrt(diff);
    EXPECT_NEAR(diff, 0, FLT_EPSILON);
}