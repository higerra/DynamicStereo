//
// Created by yanhang on 4/7/16.
//

#include "gtest/gtest.h"
#include "../base/utility.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

const double epsilon = 1e-6;

TEST(OpenCV, HSVRange){
    Mat image(1000,1000,CV_32FC3);
    cv::RNG rng;
    rng.fill(image, RNG::UNIFORM, Scalar::all(0), Scalar::all(1));
    Mat hsvImg;
    cv::cvtColor(image, hsvImg, CV_BGR2HSV);
    vector<Mat> spl;
    cv::split(hsvImg, spl);
    EXPECT_EQ(spl.size(),3);
    vector<double> minv(3), maxv(3);
    vector<double> range{360,1,1};
    for(auto c=0; c<3; ++c) {
        cv::minMaxIdx(spl[c], &minv[c], &maxv[c]);
        printf("Channel %d, min %.2f, max %.2f\n", c, minv[c], maxv[c]);
        EXPECT_GT(maxv[c], range[c]/2);
        EXPECT_LE(maxv[c], range[c]);
        EXPECT_GE(minv[c], 0);
    }
}