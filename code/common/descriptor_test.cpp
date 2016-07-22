//
// Created by yanhang on 5/28/16.
//

#include "gtest/gtest.h"
#include "CVdescriptor.h"

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
        rng.fill(img, RNG::UNIFORM, Scalar(-1,-1,-1), Scalar(1,1,1));
    }

//    vector<float> feat;
//    Feature::HoG3D hog;
//    hog.constructFeature(gradients, feat);
//    for(auto v: feat)
//        printf("%.3f ", v);
//    cout << endl;

    vector<KeyPoint> keypt(1);
    keypt[0].pt = cv::Point2f(w/2-1, h/2-1);
    keypt[0].octave = t / 2;
    CVHoG3D hog3d(100, 24);
    Mat descriptor;
    hog3d.compute(gradients, keypt, descriptor);
    for(auto i=0; i<descriptor.cols; ++i)
        printf("%.3f ", descriptor.at<float>(0,i));
    cout << endl;
}