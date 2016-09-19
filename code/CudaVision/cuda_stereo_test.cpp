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
    Vector4d spt = Vector4d::Random();
    spt[3] = 1.0;

    const double focal = 600;
    const int px = 300, py = 400;
    const double r1 = 0.01, r2 = 0.001;
    Vector3d position(1,1,1);
    Vector3d axis(0.3,0.7,0.4);
    //generate a theia camera, and convert to CudaCamera
    theia::CameraIntrinsicsPrior cam_prior;
    cam_prior.camera_intrinsics_model_type = "PINHOLE";
    cam_prior.focal_length.value[0] = focal;
    cam_prior.image_width = px * 2;
    cam_prior.image_height = py * 2;
    cam_prior.principal_point.value[0] = px;
    cam_prior.principal_point.value[1] = py;
    cam_prior.aspect_ratio.value[0] = 1.0;
    cam_prior.skew.value[0] = 0.0;
    cam_prior.radial_distortion.value[0] = r1;
    cam_prior.radial_distortion.value[1] = r2;


    theia::Camera cam;
    cam.SetFromCameraIntrinsicsPriors(cam_prior);
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

//            CHECK_EQ(cam.GetCameraIntrinsicsModelType(), theia::CameraIntrinsicsModelType::PINHOLE);
        const double* intParam = cam.intrinsics();
        intrinsic[0] = (TCam) intParam[theia::PinholeCameraModel::FOCAL_LENGTH];
        intrinsic[1] = (TCam) intParam[theia::PinholeCameraModel::ASPECT_RATIO];
        intrinsic[2] = (TCam) intParam[theia::PinholeCameraModel::SKEW];
        intrinsic[3] = (TCam) intParam[theia::PinholeCameraModel::PRINCIPAL_POINT_X];
        intrinsic[4] = (TCam) intParam[theia::PinholeCameraModel::PRINCIPAL_POINT_Y];
        intrinsic[5] = (TCam) intParam[theia::PinholeCameraModel::RADIAL_DISTORTION_1];
        intrinsic[6] = (TCam) intParam[theia::PinholeCameraModel::RADIAL_DISTORTION_2];
    };


    CudaVision::CudaCamera<double> cudacam;
    copyCamera(cam, cudacam.intrinsic, cudacam.extrinsic);

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
