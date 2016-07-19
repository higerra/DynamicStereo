//
// Created by yanhang on 5/28/16.
//

#include "gtest/gtest.h"
#include "descriptor.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

TEST(FeatureModule, HoG3D){
    //generate random 'video' gradient
    int w = 100, h = 100, t = 24;
    vector<Mat> gradients((size_t) t);
    cv::RNG rng((uint)time(NULL));
    for(auto& img: gradients){
        img.create(h,w,CV_32FC3);

    }

    vector<float> feat;
    Feature::HoG3D hog;
    hog.constructFeature(gradients, feat);
    for(auto v: feat)
        printf("%.3f ", v);
    cout << endl;
}