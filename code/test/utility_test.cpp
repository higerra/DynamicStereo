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

TEST(Math, matrix2Nomr){
    Matrix3d m;
    m << 2,0,1,-1,1,0,-3,3,0;
    cout << m << endl;
    double l2 = math_util::matrix2Norm<Matrix3d>(m);
    printf("res: %.4f\n", l2);
}