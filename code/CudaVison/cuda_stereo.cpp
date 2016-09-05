//
// Created by yanhang on 9/5/16.
//

#include "cuda_stereo.h"

#include <cmath>

namespace dynamic_stereo{

    void RadialUndistortPoint(const float distorted_point_x,
                              const float distorted_point_y,
                              const float radial_distortion1,
                              const float radial_distortion2,
                              float* undistorted_point_x,
                              float* undistorted_point_y){
//        const float kMinRadius = 1e-5;
//        const float kEpsilon = 1e-8;
//        const int kMaxIter = 10;
//
//        if(max(abs(distorted_point_x), abs(distorted_point_y)) < kMinRadius){
//            *undistorted_point_x = distorted_point_x;
//            *undistorted_point_y = distorted_point_y;
//            return;
//        }
//
//        float quintic_polynomial[6] = {};
//        if(abs(distorted_point_x) > abs(distorted_point_y)){
//            const float point_ratio = distorted_point_y / distorted_point_x;
//            const float r_sq = 1 + point_ratio * point_ratio;
//            quintic_polynomial[0] = radial_distortion2 * r_sq * r_sq;
//            quintic_polynomial[1] = 0;
//            quintic_polynomial[2] = radial_distortion1 * r_sq;
//            quintic_polynomial[3] = 0;
//            quintic_polynomial[4] = 1;
//            quintic_polynomial[5] = -distorted_point_x;
//
//            undistorted_point_x = FindRootIterativeLaguerre(
//                    quintic_polynomial, distorted_point.x(), kEpsilon, kMaxIter);
//            undistorted_point_y = point_ratio * undistorted_point->x();
//
//        }

    }

    void CudaCamera::projectPoint(const float* pt, float* pixel) {
        //transform to camera coordinate
        float cameraPt[4] = {};
        for (int y = 0; y < 4; ++y) {
            cameraPt[y] = extrinsic[y * 4] * pt[0] + extrinsic[y * 4 + 1] * pt[1] +
                          extrinsic[y * 4 + 2] * pt[2] + extrinsic[y * 4 + 3] * pt[3];
        }
        if (cameraPt[3] != 0) {
            for (int i = 0; i < 3; ++i)
                cameraPt[i] /= cameraPt[3];
        }

        const float &depth = cameraPt[2];
        const float normalized_pixel[2] = {cameraPt[0] / depth, cameraPt[1] / depth};

        float distorted_pixel[2];
        RadialDistortPoint(normalized_pixel[0], normalized_pixel[1],
                          intrinsic[RADIAL_DISTORTION_1], intrinsic[RADIAL_DISTORTION_2],
                          distorted_pixel, distorted_pixel + 1);

        // Apply calibration parameters to transform normalized units into pixels.
        const float& focal_length = intrinsic[FOCAL_LENGTH];
        const float& skew = intrinsic[SKEW];
        const float& aspect_ratio = intrinsic[ASPECT_RATIO];
        const float& principal_point_x = intrinsic[PRINCIPAL_POINT_X];
        const float& principal_point_y = intrinsic[PRINCIPAL_POINT_Y];

        pixel[0] = focal_length * distorted_pixel[0] + skew * distorted_pixel[1] +
                   principal_point_x;
        pixel[1] = focal_length * aspect_ratio * distorted_pixel[1] +
                   principal_point_y;
    }
}//namespace dynamic_stereo