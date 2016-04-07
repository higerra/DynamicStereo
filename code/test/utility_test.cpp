//
// Created by yanhang on 4/7/16.
//

#include "gtest/gtest.h"
#include "../base/utility.h"

using namespace Eigen;
TEST(Geometry, distanceToLineSegment){
    double epsilon = 1e-6;
    Vector2d spt2d(0,0);
    Vector2d ept2d(0,1);
    Vector2d p2d1(-1,1);
    Vector2d p2d2(0,1);
    Vector2d p2d3(0.5,1);
    Vector2d p2d4(10,1);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d1,spt2d,ept2d), std::sqrt(2.0), epsilon);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d2,spt2d,ept2d), 1, epsilon);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d3,spt2d,ept2d), 0.5, epsilon);
    EXPECT_NEAR(geometry_util::distanceToLineSegment<2>(p2d4,spt2d,ept2d), std::sqrt(82), epsilon);
}