//
// Created by yanhang on 9/5/16.
//

#include "cuStereoUtil.h"
#include "gtest/gtest.h"
#include <theia/theia.h>
#include <Eigen/Eigen>
#include "../base/utility.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

using namespace Eigen;
using namespace std;
using namespace cv;

TEST(CudaStereo, RadialDistortion){
    const Vector2d undistorted_point = Vector2d::Random();
    Vector2d distorted_point, distorted_point_theia;
    double k1 = 0, k2 = 0;
    CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);
    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());

    k1 = 0.1; k2 = 0;
    CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());

    k1 = 0.1, k2 = 0.05;
    CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                       k1, k2, distorted_point.data(), distorted_point.data() + 1);

    theia::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
                              k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

    EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());
}

TEST(CudaStereo, ProjectPointToImage){
    Vector4d spt = Vector4d::Random();
    spt[3] = 1.0;

    const double focal = 600;
    const int px = 300, py = 400;
    const double r1 = 0.01, r2 = 0.001;
    Vector3d position(1,1,1);
    Vector3d axis(0.3,0.7,0.4);
    //generate a theia camera, and convert to CudaCamera
    theia::Camera cam;
    cam.SetFocalLength(focal);
    cam.SetPrincipalPoint(px, py);
    cam.SetImageSize(px*2, py*2);
    cam.SetAspectRatio(1.0);
    cam.SetRadialDistortion(r1, r2);
    cam.SetPosition(position);
    cam.SetOrientationFromAngleAxis(axis);

    CudaVision::CudaCamera<double> cudacam;
    {
        using namespace CudaVision;
        cudacam.intrinsic[CudaCamera<double>::FOCAL_LENGTH] = focal;
        cudacam.intrinsic[CudaCamera<double>::PRINCIPAL_POINT_X] = px;
        cudacam.intrinsic[CudaCamera<double>::PRINCIPAL_POINT_Y] = py;
        cudacam.intrinsic[CudaCamera<double>::RADIAL_DISTORTION_1] = r1;
        cudacam.intrinsic[CudaCamera<double>::RADIAL_DISTORTION_2] = r2;
        cudacam.intrinsic[CudaCamera<double>::ASPECT_RATIO] = 1.0;
        cudacam.extrinsic[CudaCamera<double>::ORIENTATION] = axis[0];
        cudacam.extrinsic[CudaCamera<double>::ORIENTATION+1] = axis[1];
        cudacam.extrinsic[CudaCamera<double>::ORIENTATION+2] = axis[2];
        cudacam.extrinsic[CudaCamera<double>::POSITION] = position[0];
        cudacam.extrinsic[CudaCamera<double>::POSITION + 1] = position[1];
        cudacam.extrinsic[CudaCamera<double>::POSITION + 2] = position[2];
    }


    Vector2d pixel_cuda, pixel_theia;
    cam.ProjectPoint(spt, &pixel_theia);
    cudacam.projectPoint(spt.data(), pixel_cuda.data());

    printf("test pt: (%.3f,%.3f,%.3f)\n", spt[0], spt[1], spt[2]);
    printf("Theia: (%.3f,%.3f), Cuda: (%.3f,%.3f)\n", pixel_theia[0], pixel_theia[1], pixel_cuda[0], pixel_cuda[1]);
    EXPECT_NEAR((pixel_cuda - pixel_theia).norm(), 0.0, DBL_EPSILON);
}

TEST(CudaStereo, bilinear) {
    Mat ranMat(10, 10, CV_8UC3);
    cv::RNG rng;
    rng.fill(ranMat, RNG::UNIFORM, Scalar(0, 0, 0), Scalar(255, 255, 255));

    std::default_random_engine engine;
    std::uniform_real_distribution<double> dist(1, 9);
    for (int k = 0; k < 5; ++k) {
        double loc[2]{dist(engine), dist(engine)};
        double pixCuda[3];
        CudaVision::bilinearInterpolation<unsigned char, double, double>(ranMat.data, ranMat.cols, loc, pixCuda);
        Vector3d pixUtility = interpolation_util::bilinear<unsigned char, 3>(ranMat.data, ranMat.cols, ranMat.rows,
                                                                             Eigen::Map<Vector2d>(loc));

        printf("pt:(%.3f,%.3f), utility:(%.3f,%.3f,%.3f), cuda:(%.3f,%.3f,%.3f)\n", loc[0], loc[1],
               pixUtility[0], pixUtility[1], pixUtility[2], pixCuda[0], pixCuda[1], pixCuda[2]);
        EXPECT_NEAR((pixUtility - Eigen::Map<Eigen::Vector3d>(pixCuda) ).norm(), 0.0, 0.1);
    }
}