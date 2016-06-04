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

TEST(Geometry_distanceToLineSegment, inside){
    Vector2d spt2d(0,0);
    Vector2d ept2d(1,0);
    Vector2d p2d1(0.5,1);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d1,spt2d,ept2d), 1.0, epsilon);
}

TEST(Geometry_distanceToLineSegment, outside){
    double epsilon = 1e-6;
    Vector2d spt2d(0,0);
    Vector2d ept2d(1,0);
    Vector2d p2d1(-1,1);
    Vector2d p2d2(0,1);
    Vector2d p2d3(10,1);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d1,spt2d,ept2d), std::sqrt(2.0), epsilon);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d2,spt2d,ept2d), 1, epsilon);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d3,spt2d,ept2d), std::sqrt(82), epsilon);
}

TEST(Misc_OpenCV, norm){
    Vec3b pix1(100,100,100);
    Vec3b pix2(0,0,0);
    Vector3d pix1v((double)pix1[0], (double)pix1[1], (double)pix1[2]);
    Vector3d pix2v((double)pix2[0], (double)pix2[1], (double)pix2[2]);
    EXPECT_NEAR(cv::norm(pix1-pix2), (pix1v-pix2v).norm(), epsilon);
}

TEST(Misc_OpenCV, VecCast){
    Vec3b x(10,20,30);
    Vec3f y1 = static_cast<Vec3f>(x);
    Vec3f y2(10,20,30);
    EXPECT_DOUBLE_EQ(cv::norm(y1-y2), 0.0);
}