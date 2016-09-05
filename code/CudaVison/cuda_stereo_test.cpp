//
// Created by yanhang on 9/5/16.
//

#include "cuda_stereo.h"
#include "gtest/gtest.h"
#include <theia/theia.h>
#include <Eigen/Eigen>
#include <iostream>

using namespace dynamic_stereo;
using namespace Eigen;
using namespace std;

TEST(CudaStereo, RadialDistortion){
    const Vector2f undistorted_point = Vector2f::Random();

    cout << "Undistorted point: " << undistorted_point.transpose() << endl;

    Vector2f distorted_point, distorted_point_theia;
    float k1 = 0, k2 = 0;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);
    printf("k1: %.3f, k2: %.3f\n", k1, k2);
    cout << "Distorted cuda: " << distorted_point.transpose() << endl;
    cout << "Distorted theia: " << distorted_point_theia.transpose() << endl;
    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());

    k1 = 0.1; k2 = 0;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    printf("k1: %.3f, k2: %.3f\n", k1, k2);
    cout << "Distorted cuda: " << distorted_point.transpose() << endl;
    cout << "Distorted theia: " << distorted_point_theia.transpose() << endl;
    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());

    k1 = 0.1, k2 = 0.05;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    printf("k1: %.3f, k2: %.3f\n", k1, k2);
    cout << "Distorted cuda: " << distorted_point.transpose() << endl;
    cout << "Distorted theia: " << distorted_point_theia.transpose() << endl;
    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());
}

TEST(CudaStereo, ProjectPointToImage){

}