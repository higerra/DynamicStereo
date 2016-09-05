//
// Created by yanhang on 9/5/16.
//

#ifndef DYNAMICSTEREO_CUDA_STEREO_H
#define DYNAMICSTEREO_CUDA_STEREO_H

namespace dynamic_stereo{
   inline void RadialDistortPoint(const float undistorted_point_x,
                            const float undistorted_point_y,
                            const float radial_distortion1,
                            const float radial_distortion2,
                            float* distorted_point_x,
                            float* distorted_point_y){
        const float r_sq = undistorted_point_x * undistorted_point_x + undistorted_point_y * undistorted_point_y;
        const float d = 1.0f + r_sq * (radial_distortion1 + radial_distortion2 * r_sq);
        (*distorted_point_x) = undistorted_point_x * d;
        (*distorted_point_y) = undistorted_point_y * d;
    }

    void RadialUndistortPoint(const float distorted_point_x,
                              const float distorted_point_y,
                              const float radial_distortion1,
                              const float radial_distortion2,
                              float* undistorted_point_x,
                              float* undistorted_point_y);

    struct CudaCamera{
        enum InternalParametersIndex{
            FOCAL_LENGTH = 0,
            ASPECT_RATIO = 1,
            SKEW = 2,
            PRINCIPAL_POINT_X = 3,
            PRINCIPAL_POINT_Y = 4,
            RADIAL_DISTORTION_1 = 5,
            RADIAL_DISTORTION_2 = 6
        };

        static const int kExtrinsicSize = 16;
        static const int kIntrinsicSize = 7;

        //intrinsic parameters: focal_length, skew, aspect_ratio, px, py
        float intrinsic[kIntrinsicSize];
        float extrinsic[kExtrinsicSize];

        void projectPoint(const float* pt, float* pixel);

    };



}


#endif //DYNAMICSTEREO_CUDA_STEREO_H
