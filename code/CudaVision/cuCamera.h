//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUCAMERA_H
#define DYNAMICSTEREO_CUCAMERA_H

#include <cmath>
#include <cfloat>
#include <stdlib.h>

namespace CudaVision{
    template <typename T>
    __device__ __host__ void RadialDistortPoint(const T undistorted_point_x,
                                   const T undistorted_point_y,
                                   const T radial_distortion1,
                                   const T radial_distortion2,
                                   T* distorted_point_x,
                                   T* distorted_point_y){
        const T r_sq = undistorted_point_x * undistorted_point_x + undistorted_point_y * undistorted_point_y;
        const T d = 1.0f + r_sq * (radial_distortion1 + radial_distortion2 * r_sq);
        (*distorted_point_x) = undistorted_point_x * d;
        (*distorted_point_y) = undistorted_point_y * d;
    }


    //re-implement void AngleAxisRotatePoint() from Ceres
    template<typename T>
    __device__ __host__ void angleAxisRotatePoint(const T axis_angle[], const T pt[3], T result[3]) {
        const T theta2 =
                axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] + axis_angle[2] * axis_angle[2];
        if (theta2 > (T)FLT_EPSILON) {
            const T theta = sqrt(theta2);
            const T costheta = cos(theta);
            const T sintheta = sin(theta);
            const T theta_inverse = 1.0 / theta;
            const T w[3] = {axis_angle[0] * theta_inverse,
                            axis_angle[1] * theta_inverse,
                            axis_angle[2] * theta_inverse};

            const T w_cross_pt[3] = {w[1] * pt[2] - w[2] * pt[1],
                                     w[2] * pt[0] - w[0] * pt[2],
                                     w[0] * pt[1] - w[1] * pt[0]};

            const T tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1.0f - costheta);

            result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
            result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
            result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
        }else{
            const T w_cross_pt[3] = {axis_angle[1] * pt[2] - axis_angle[2] * pt[1],
                                     axis_angle[2] * pt[0] - axis_angle[0] * pt[2],
                                     axis_angle[0] * pt[1] - axis_angle[1] * pt[0]};
            result[0] = pt[0] + w_cross_pt[0];
            result[1] = pt[1] + w_cross_pt[1];
            result[2] = pt[2] + w_cross_pt[2];
        }
    }

    //camera object should be initalized in CPU and copied into GPU
    template<typename T = double>
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
        enum ExternalParametersIndex{
            POSITION = 0,
            ORIENTATION = 3
        };
        __device__ __host__ void setOrientationFromAngleAxis(const T v1, const T v2, const T v3){
            extrinsic[ORIENTATION] = v1;
            extrinsic[ORIENTATION+1] = v2;
            extrinsic[ORIENTATION+2] = v3;
        }

        __device__ __host__ void setPosition(const T x, const T y, const T z){
            extrinsic[POSITION] = x;
            extrinsic[POSITION + 1] = y;
            extrinsic[POSITION + 2] = z;
        }

        __device__ __host__ const T* getPosition() const{return extrinsic + POSITION; }
        __device__ __host__ const T* getOrientation() const{return extrinsic + ORIENTATION; }

        static const int kExtrinsicSize = 6;
        static const int kIntrinsicSize = 7;

        //intrinsic parameters: focal_length, skew, aspect_ratio, px, py
        T intrinsic[kIntrinsicSize];
        T extrinsic[kExtrinsicSize];
        __device__ __host__ T projectPoint(const T* pt, T* pixel);
    };

    template<typename T>
    __device__ __host__ T CudaCamera<T>::projectPoint(const T* pt, T* pixel) {
        //transform to camera coordinate
        T adjusted_point[3];
        for(int i=0; i<3; ++i)
            adjusted_point[i] = pt[i] - pt[3] * extrinsic[POSITION+i];

        T rotated_point[3];
        angleAxisRotatePoint(extrinsic + ORIENTATION, adjusted_point, rotated_point);

        const T &depth = rotated_point[2];
        const T normalized_pixel[2] = {rotated_point[0] / depth, rotated_point[1] / depth};

        T distorted_pixel[2];
        RadialDistortPoint(normalized_pixel[0], normalized_pixel[1],
                           intrinsic[RADIAL_DISTORTION_1], intrinsic[RADIAL_DISTORTION_2],
                           distorted_pixel, distorted_pixel + 1);

        // Apply calibration parameters to transform normalized units into pixels.
        const T& focal_length = intrinsic[FOCAL_LENGTH];
        const T& skew = intrinsic[SKEW];
        const T& aspect_ratio = intrinsic[ASPECT_RATIO];
        const T& principal_point_x = intrinsic[PRINCIPAL_POINT_X];
        const T& principal_point_y = intrinsic[PRINCIPAL_POINT_Y];

        pixel[0] = focal_length * distorted_pixel[0] + skew * distorted_pixel[1] +
                   principal_point_x;
        pixel[1] = focal_length * aspect_ratio * distorted_pixel[1] +
                   principal_point_y;

        return depth / pt[3];
    }
}//namespace CudaVision
#endif //DYNAMICSTEREO_CUCAMERA_H
