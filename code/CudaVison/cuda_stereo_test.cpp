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
    Vector2f distorted_point, distorted_point_theia;
    float k1 = 0, k2 = 0;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);
    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());

    k1 = 0.1; k2 = 0;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());

    k1 = 0.1, k2 = 0.05;
    RadialDistortPoint(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<float>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<float>::min());
}

TEST(CudaStereo, ProjectPointToImage){
    float k1 = 0.1, k2 = 0.05;
    Vector4f spt = Vector4f::Random();
    spt[3] = 1.0;

    //generate a theia camera, and convert to CudaCamera
    theia::Camera cam;
    cam.SetFocalLength(600);
    cam.SetPrincipalPoint(300, 400);
    cam.SetRadialDistortion(0.01, 0.001);
    cam.SetPosition(Vector3d::Ones());
    Vector3d gt_angle_axis(0.3,0.7,0.4);
    Matrix3d gt_rotation_matrix;
    ceres::AngleAxisToRotationMatrix(gt_angle_axis.data(), gt_rotation_matrix.data());
    cam.SetOrientationFromRotationMatrix(gt_rotation_matrix);

    CudaCamera cudacam;
    theia::Matrix3x4d extrinsic_theia;
    cam.GetProjectionMatrix(&extrinsic_theia);

    Vector2d pixel_cuda, pixel_theia;

}