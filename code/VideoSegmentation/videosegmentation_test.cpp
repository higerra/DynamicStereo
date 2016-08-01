//
// Created by yanhang on 7/31/16.
//

#include "gtest/gtest.h"
#include "videosegmentation.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

//TEST(Feature, BRIEFWrapper){
//    shared_ptr<BRIEFWrapper> brief(new BRIEFWrapper());
//    Mat image = imread("brief_test.png");
//    CHECK(image.data);
//
//    float start_t = (float)cv::getTickCount();
//    vector<vector<uchar> > feats;
//    brief->extractImage(image, feats);
//    printf("Time: %.3fs\n", ((float)getTickCount() - start_t) / (float)cv::getTickFrequency());
//    EXPECT_EQ(feats.size(), image.rows * image.cols);
//    EXPECT_EQ(feats[0].size(), 32);
//}

TEST(Distance, Hamming){
    vector<uchar> a1{0b0000, 0b0001, 0b0011};
    vector<uchar> a2{0b0001, 0b0001, 0b0011};
    int res =  (int)norm(a1, a2, cv::NORM_HAMMING);
    EXPECT_EQ(res, 1);
}