//
// Created by yanhang on 9/5/16.
//
#include "cuda_stereo.h"
namespace CudaVision{
    void bilinearInterpolation(const unsigned char* const data, const int w,
                               const double loc[2], float res[3]){
        const float epsilon = 0.00001;
        int xl = floor(loc[0] - epsilon), xh = (int) round(loc[0] + 0.5 - epsilon);
        int yl = floor(loc[1] - epsilon), yh = (int) round(loc[1] + 0.5 - epsilon);

        if (loc[0] <= epsilon)
            xl = 0;
        if (loc[1] <= epsilon)
            yl = 0;

        const int l1 = yl * w + xl;
        const int l2 = yh * w + xh;
        if (l1 == l2) {
            for (size_t i = 0; i < 3; ++i)
                res[i] = (float)data[l1 * 3 + i];
            return;
        }

        float lm = loc[0] - (float) xl, rm = (float) xh - loc[0];
        float tm = loc[1] - (float) yl, bm = (float) yh - loc[1];
        int ind[4] = {xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w};

        float v[4][3] = {};
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 3; ++j)
                v[i][j] = (float)data[ind[i] * 3 + j];
        }
        if (fabs(lm) <= epsilon && fabs(rm) <= epsilon){
            res[0] = (v[0][0] * bm + v[2][0] * tm) / (bm + tm);
            res[1] = (v[0][1] * bm + v[2][1] * tm) / (bm + tm);
            res[2] = (v[0][2] * bm + v[2][2] * tm) / (bm + tm);
            return;
        }

        if (fabs(bm) <= epsilon && fabs(tm) <= epsilon) {
            res[0] = (v[0][0] * rm + v[2][0] * lm) / (lm + rm);
            res[1] = (v[0][1] * rm + v[2][1] * lm) / (lm + rm);
            res[2] = (v[0][2] * rm + v[2][2] * lm) / (lm + rm);
            return;
        }

        float vw[4] = {rm * bm, lm * bm, lm * tm, rm * tm};
        float sum = vw[0] + vw[1] + vw[2] + vw[3];

        res[0] = (v[0][0] * vw[0] + v[1][0] * vw[1] + v[2][0] * vw[2] + v[3][0] * vw[3]) / sum;
        res[1] = (v[0][1] * vw[0] + v[1][1] * vw[1] + v[2][1] * vw[2] + v[3][1] * vw[3]) / sum;
        res[2] = (v[0][2] * vw[0] + v[1][2] * vw[1] + v[2][2] * vw[2] + v[3][2] * vw[3]) / sum;
    }



    double CudaCamera::projectPoint(const double* pt, double* pixel) {
        //transform to camera coordinate
        double adjusted_point[3];
        for(int i=0; i<3; ++i)
            adjusted_point[i] = pt[i] - pt[3] * extrinsic[POSITION+i];

        double rotated_point[3];
        angleAxisRotatePoint(extrinsic + ORIENTATION, adjusted_point, rotated_point);

        const double &depth = rotated_point[2];
        const double normalized_pixel[2] = {rotated_point[0] / depth, rotated_point[1] / depth};

        double distorted_pixel[2];
        RadialDistortPoint(normalized_pixel[0], normalized_pixel[1],
                          intrinsic[RADIAL_DISTORTION_1], intrinsic[RADIAL_DISTORTION_2],
                          distorted_pixel, distorted_pixel + 1);

        // Apply calibration parameters to transform normalized units into pixels.
        const double& focal_length = intrinsic[FOCAL_LENGTH];
        const double& skew = intrinsic[SKEW];
        const double& aspect_ratio = intrinsic[ASPECT_RATIO];
        const double& principal_point_x = intrinsic[PRINCIPAL_POINT_X];
        const double& principal_point_y = intrinsic[PRINCIPAL_POINT_Y];

        pixel[0] = focal_length * distorted_pixel[0] + skew * distorted_pixel[1] +
                   principal_point_x;
        pixel[1] = focal_length * aspect_ratio * distorted_pixel[1] +
                   principal_point_y;

        return depth / pt[3];
    }
}//namespace CudaVision