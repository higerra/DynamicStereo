//
// Created by yanhang on 9/20/16.
//

#include "warping.h"
#include "tracking.h"
#include "factorization.h"

#include <iostream>
#include <ctime>
#include <random>
#include <gtest/gtest.h>

#include <Eigen/Eigen>

using namespace substab;
using namespace cv;
using namespace std;

TEST(Math, SolveInverseBilinear){

}

TEST(Grid, warpAndInverseWarp){
    const int imgDim = 1024, gridDim = 64;
    const double maxOffset = 100;
    const int kPt = 50;
    GridWarpping testGrid(imgDim, imgDim, gridDim, gridDim);
    vector<Eigen::Vector2d> srcPt(kPt), tgtPt(kPt);

    std::default_random_engine engine((unsigned long)time(NULL));
    std::uniform_real_distribution<double> distPt(0, imgDim - 1);
    std::uniform_real_distribution<double> distOffset(-maxOffset, maxOffset);
    for(auto i=0; i<kPt; ++i){
        tgtPt[i][0] = distPt(engine);
        tgtPt[i][1] = distPt(engine);
        srcPt[i][0]= tgtPt[i][0] + distOffset(engine);
        srcPt[i][1]= tgtPt[i][1] + distOffset(engine);
    }

    testGrid.computeWarpingField(srcPt, tgtPt);
    const int kTest = 10;
    for(auto i=0; i<kTest; ++i){
        Eigen::Vector2d testPt;
        testPt[0] = distPt(engine);
        testPt[1] = distPt(engine);
        Eigen::Vector2d warped = testGrid.warpPoint(testPt);
        Eigen::Vector2d rewarp;
        bool sucess = testGrid.inverseWarpPoint(warped, rewarp);
        if(sucess) {
            LOG(INFO) << "Solution found for test point: " << testPt.transpose() << '\t' << rewarp.transpose();
            EXPECT_NEAR((testPt - rewarp).norm(), 0.0, 0.001);
        } else {
            LOG(WARNING) << "No solution: " << testPt.transpose();
        }
    }
}
