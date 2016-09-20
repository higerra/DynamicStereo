//
// Created by yanhang on 9/5/16.
//


#include "cuStereoUtil.h"
#include "cuCamera.h"
#include <theia/theia.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include "gtest/gtest.h"
#include "../base/utility.h"

using namespace Eigen;
using namespace std;
using namespace cv;

// TEST(CudaStereo, RadialDistortion){
//     const Vector2d undistorted_point = Vector2d::Random();
//     Vector2d distorted_point, distorted_point_theia;
//     double k1 = 0, k2 = 0;
//     CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 					   k1, k2, distorted_point.data(), distorted_point.data() + 1);

// theia::PinholeCameraModel::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 				      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);
//     EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());

//     k1 = 0.1; k2 = 0;
//     CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 					   k1, k2, distorted_point.data(), distorted_point.data() + 1);

// theia::PinholeCameraModel::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 				      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

//     EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());

//     k1 = 0.1, k2 = 0.05;
//     CudaVision::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 					   k1, k2, distorted_point.data(), distorted_point.data() + 1);

// theia::PinholeCameraModel::RadialDistortPoint<double>(undistorted_point.x(), undistorted_point.y(),
// 				      k1, k2, distorted_point_theia.data(), distorted_point_theia.data() + 1);

//     EXPECT_NEAR((distorted_point - distorted_point_theia).norm(), 0.0, std::numeric_limits<double>::min());
// }

TEST(CudaStereo, ProjectPointToImage){
    Vector4d spt(-381.556,26.472,213.060,1.0);

    const double focal = 1066.53;
    const double px = 821.791, py = 1021.12;
    const double r1 = -0.0354408, r2 = -0.00696851;
    Vector3d position(-31.233,6.488,-20.537);
    Vector3d axis(0.082,0.432,-0.044);
    //generate a theia camera, and convert to CudaCamera

    theia::Camera cam;
    cam.SetFocalLength(focal);
    cam.SetPrincipalPoint(px, py);
    cam.SetSkew(0.0);
    cam.SetAspectRatio(1.0);
    cam.SetRadialDistortion(r1, r2);

    cam.SetPosition(position);
    cam.SetOrientationFromAngleAxis(axis);

    using TCam = double;
    auto copyCamera = [](const theia::Camera &cam, TCam *intrinsic, TCam *extrinsic) {
        Vector3d pos = cam.GetPosition();
        Vector3d ax = cam.GetOrientationAsAngleAxis();
        for (auto i = 0; i < 3; ++i) {
            extrinsic[i] = pos[i];
            extrinsic[i + 3] = ax[i];
        }
        intrinsic[0] = (TCam) cam.FocalLength();
        intrinsic[1] = (TCam) cam.AspectRatio();
        intrinsic[2] = (TCam) cam.Skew();
        intrinsic[3] = (TCam) cam.PrincipalPointX();
        intrinsic[4] = (TCam) cam.PrincipalPointY();
        intrinsic[5] = (TCam) cam.RadialDistortion1();
        intrinsic[6] = (TCam) cam.RadialDistortion2();
    };


    CudaVision::CudaCamera<double> cudacam;
    copyCamera(cam, cudacam.intrinsic, cudacam.extrinsic);

    printf("Cuda cam:\nIntrinsic:");
    for(auto i=0; i<cudacam.kIntrinsicSize; ++i)
        cout << cudacam.intrinsic[i] << ' ';
    cout << endl << "Extrinsic:";
    for(auto i=0; i<cudacam.kExtrinsicSize; ++i)
        cout << cudacam.extrinsic[i] << ' ';
    cout << endl;

    Vector2d pixel_cuda, pixel_theia;
    cam.ProjectPoint(spt, &pixel_theia);
    cudacam.projectPoint(spt.data(), pixel_cuda.data());

    printf("test pt: (%.3f,%.3f,%.3f)\n", spt[0], spt[1], spt[2]);
    printf("Theia: (%.3f,%.3f), Cuda: (%.3f,%.3f)\n", pixel_theia[0], pixel_theia[1], pixel_cuda[0], pixel_cuda[1]);
    EXPECT_NEAR((pixel_cuda - pixel_theia).norm(), 0.0, DBL_EPSILON);
}

//TEST(CudaStereo, bilinear) {
//    Mat ranMat(10, 10, CV_8UC3);
//    cv::RNG rng;
//    rng.fill(ranMat, RNG::UNIFORM, Scalar(0, 0, 0), Scalar(255, 255, 255));
//
//    std::default_random_engine engine;
//    std::uniform_real_distribution<double> dist(1, 9);
//    for (int k = 0; k < 5; ++k) {
//        double loc[2]{dist(engine), dist(engine)};
//        double pixCuda[3];
//        CudaVision::bilinearInterpolation<unsigned char, double, double>(ranMat.data, ranMat.cols, loc, pixCuda);
//        Vector3d pixUtility = interpolation_util::bilinear<unsigned char, 3>(ranMat.data, ranMat.cols, ranMat.rows,
//                                                                             Eigen::Map<Vector2d>(loc));
//
//        printf("pt:(%.3f,%.3f), utility:(%.3f,%.3f,%.3f), cuda:(%.3f,%.3f,%.3f)\n", loc[0], loc[1],
//               pixUtility[0], pixUtility[1], pixUtility[2], pixCuda[0], pixCuda[1], pixCuda[2]);
//        EXPECT_NEAR((pixUtility - Eigen::Map<Eigen::Vector3d>(pixCuda) ).norm(), 0.0, 0.1);
//    }
//}

TEST(CudaUtil, find_nth){
    vector<float> array(10);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(1, 9);

    for(auto i=0; i<array.size(); ++i){
        array[i] = dist(engine);
    }

    vector<float> array_ori = array;

    const int kth = array.size() / 2;
    nth_element(array.begin(), array.begin() + kth, array.end());

    float kth_res = CudaVision::find_nth<float>(array_ori.data(), array_ori.size(), kth);
    EXPECT_NEAR(array[kth], kth_res, FLT_EPSILON);
}


TEST(CudaUtil, quick_sort){
    vector<int> array(20);

    std::default_random_engine engine;
    std::uniform_int_distribution<int> dist(0, 100);
    for(auto i=0; i<array.size(); ++i)
        array[i] = dist(engine);

    printf("Before sort:\n");
    for(auto v: array)
        cout << v << ' ';
    cout << endl;

    vector<int> array_quick = array;
    vector<int> array_insert = array;
    std::sort(array.begin(), array.end());
    CudaVision::quick_sort<int>(array_quick.data(), 0, (int)array.size() - 1);
    CudaVision::insert_sort<int>(array_insert.data(), (int)array.size());

    printf("After sort:\n");
    printf("STL:");
    for(auto v: array)
        cout << v << ' ';
    cout << endl;

    printf("CUQ:");
    for(auto v: array_quick)
        cout << v << ' ';
    cout << endl;

    printf("CUI:");
    for(auto v: array_quick)
        cout << v << ' ';
    cout << endl;

    int diff = 0;
    for(auto i=0; i<array.size(); ++i){
        diff += array[i] - array_quick[i];
    }
    EXPECT_EQ(diff, 0);

    diff = 0;
    for(auto i=0; i<array.size(); ++i){
        diff += array[i] - array_insert[i];
    }
    EXPECT_EQ(diff, 0);
}
