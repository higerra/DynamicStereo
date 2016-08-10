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

TEST(OpenCV, horizontalIntegral){
    Mat data(3, 10, CV_32FC1, Scalar::all(1.0f));
    Mat inte(3, 11, CV_32FC1, Scalar::all(0.0f));
    for(auto i=1; i<inte.cols; ++i)
        inte.col(i) = inte.col(i-1) + data.col(i-1);
    cout << "Data: " << endl << data << endl;
    cout << "Horizontal integral: " << endl << inte << endl;

    for(auto sid=0; sid < 5; sid+=2){
        for(auto eid=5; eid<10; eid+=2){
            printf("sid: %d, eid:%d\n", sid, eid);
            Mat mv = (inte.col(eid+1) - inte.col(sid)) / (float)(eid-sid+1);
            cout << data.colRange(sid, eid+1) << endl;
            cout << mv << endl;
        }
    }
}