//
// Created by yanhang on 5/30/16.
//

#include <gtest/gtest.h>
#include "classifier.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

TEST(classifier, perturbSamples){
    Mat toyMat(5,5,CV_32F,Scalar::all(0));
    cv::randu(toyMat, 0, 1);
    cout << "Original mat:" << endl << toyMat << endl;
    perturbSamples(toyMat);
    cout << "After pertub:" << endl << toyMat << endl;
}