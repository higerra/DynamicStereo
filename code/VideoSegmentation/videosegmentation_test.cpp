//
// Created by yanhang on 7/31/16.
//

#include <opencv2/xfeatures2d.hpp>
#include "gtest/gtest.h"
#include "videosegmentation.h"
#include "colorGMM.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

TEST(Eigen, replicate){
    Vector3d m = Vector3d::Random();
    Matrix<double, 10, 3> me = m.transpose().replicate(10,1);
    cout << m << endl;
    cout << me << endl;
}
