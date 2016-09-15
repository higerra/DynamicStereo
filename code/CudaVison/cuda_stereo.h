//
// Created by yanhang on 9/5/16.
//

#ifndef DYNAMICSTEREO_CUDA_STEREO_H
#define DYNAMICSTEREO_CUDA_STEREO_H
#include <cmath>
#include <cfloat>
#include <stdlib.h>
#include <iostream>

namespace CudaVision{
   inline void RadialDistortPoint(const double undistorted_point_x,
                            const double undistorted_point_y,
                            const double radial_distortion1,
                            const double radial_distortion2,
                            double* distorted_point_x,
                            double* distorted_point_y){
        const double r_sq = undistorted_point_x * undistorted_point_x + undistorted_point_y * undistorted_point_y;
        const double d = 1.0f + r_sq * (radial_distortion1 + radial_distortion2 * r_sq);
        (*distorted_point_x) = undistorted_point_x * d;
        (*distorted_point_y) = undistorted_point_y * d;
    }


    //re-implement void AngleAxisRotatePoint() from Ceres
    inline void angleAxisRotatePoint(const double axis_angle[], const double pt[3], double result[3]) {
        const double theta2 =
                axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] + axis_angle[2] * axis_angle[2];
        if (theta2 > DBL_EPSILON) {
            const double theta = sqrt(theta2);
            const double costheta = cos(theta);
            const double sintheta = sin(theta);
            const double theta_inverse = 1.0 / theta;
            const double w[3] = {axis_angle[0] * theta_inverse,
                                axis_angle[1] * theta_inverse,
                                axis_angle[2] * theta_inverse};

            const double w_cross_pt[3] = {w[1] * pt[2] - w[2] * pt[1],
                                         w[2] * pt[0] - w[0] * pt[2],
                                         w[0] * pt[1] - w[1] * pt[0]};

            const double tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1.0f - costheta);

            result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
            result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
            result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
        }else{
            const double w_cross_pt[3] = {axis_angle[1] * pt[2] - axis_angle[2] * pt[1],
                                         axis_angle[2] * pt[0] - axis_angle[0] * pt[2],
                                         axis_angle[0] * pt[1] - axis_angle[1] * pt[0]};
            result[0] = pt[0] + w_cross_pt[0];
            result[1] = pt[1] + w_cross_pt[1];
            result[2] = pt[2] + w_cross_pt[2];
        }
    }

    //bilinear interpolation running on device
    void bilinearInterpolation(const unsigned char* const data, const int w,
                                      const double loc[2], float res[3]);

    template<typename T>
    T findMedian(T* array) {
        
    }


    //camera object should be initalized in CPU and copied into GPU
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
        void setOrientationFromRotationMatrix(const double* roration){}
        inline void setOrientationFromAngleAxis(const double v1, const double v2, const double v3){
            extrinsic[ORIENTATION] = v1;
            extrinsic[ORIENTATION+1] = v2;
            extrinsic[ORIENTATION+2] = v3;
        }

        inline void setPosition(const double x, const double y, const double z){
            extrinsic[POSITION] = x;
            extrinsic[POSITION + 1] = y;
            extrinsic[POSITION + 2] = z;
        }

        static const int kExtrinsicSize = 6;
        static const int kIntrinsicSize = 7;

        //intrinsic parameters: focal_length, skew, aspect_ratio, px, py
        double intrinsic[kIntrinsicSize];
        double extrinsic[kExtrinsicSize];
        int image_size[2];

        double projectPoint(const double* pt, double* pixel);
    };



}//namespace CudaVision


#endif //DYNAMICSTEREO_CUDA_STEREO_H
