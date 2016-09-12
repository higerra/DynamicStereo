//
// Created by yanhang on 9/11/16.
//

#ifndef DYNAMICSTEREO_STAB_ENERGY_H
#define DYNAMICSTEREO_STAB_ENERGY_H

#include "base/utility.h"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

namespace dynamic_stereo {

    struct WarpFunctorDense {
    public:
        WarpFunctorDense(const cv::Mat& srcImg_, const Eigen::Vector3d& tgtColor_,
                         const Eigen::Vector4i& biInd_, const Eigen::Vector4d& biW_,
                         const std::vector<Eigen::Vector2d>& loc_, const double weight_)
                : srcImg(srcImg_), tgtColor(tgtColor_), biInd(biInd_), biW(biW_), loc(loc_), weight(std::sqrt(weight_)){}
        bool operator()(const double* const g1, const double* const g2, const double* const g3,
                        const double* const g4, double* residual) const{
            Eigen::Vector2d pt2 = (Eigen::Vector2d(g1[0], g1[1]) + loc[biInd[0]]) * biW[0] +
                    (Eigen::Vector2d(g2[0], g2[1]) + loc[biInd[1]]) * biW[1] +
                    (Eigen::Vector2d(g3[0], g3[1]) + loc[biInd[2]]) * biW[2] +
                    (Eigen::Vector2d(g4[0], g4[1]) + loc[biInd[3]]) * biW[3];
            Eigen::Vector3d srcColor = interpolation_util::bilinear<uchar,3>(srcImg.data, srcImg.cols, srcImg.rows, pt2);
            residual[0] = (srcColor - tgtColor).norm() * weight;
            return true;
        }

    private:
        const cv::Mat& srcImg;
        const Eigen::Vector3d& tgtColor;
        const Eigen::Vector4i& biInd;
        const Eigen::Vector4d& biW;
        const std::vector<Eigen::Vector2d>& loc;

        const double weight;
    };

    struct WarpFunctorFix{
    public:
        WarpFunctorFix(const std::vector<double>* fixedV_, const double weight_)
                :fixedV(fixedV_), weight(std::sqrt(weight_)){}
        template<typename T>
        bool operator()(const T* const g, T* residual) const{
            T res1, res2;
            if(fixedV){
                res1 = g[0] - (*fixedV)[0];
                res2 = g[1] - (*fixedV)[1];
            }
            residual[0] = ceres::sqrt(res1 * res1 + res2 * res2 + (T)std::numeric_limits<double>::min()) * weight;
            return true;
        }
    private:
        const std::vector<double>* fixedV;
        const double weight;
    };

    struct WarpFunctorSimilarity{
    public:
        template<typename T>
        Eigen::Matrix<T,2,1> getLocalCoord(const Eigen::Matrix<T,2,1>& p1, const Eigen::Matrix<T,2,1>& p2, const Eigen::Matrix<T,2,1>& p3)const {
            Eigen::Matrix<T,2,1> ax1 = p3 - p2;
            Eigen::Matrix<T,2,1> ax2(-1.0*ax1[1], ax1[0]);
            CHECK_GT(ax1.norm(), (T)0.0);
            Eigen::Matrix<T,2,1> uv;
            uv[0] = (p1-p2).dot(ax1) / ax1.norm() / ax1.norm();
            uv[1] = (p1-p2).dot(ax2) / ax2.norm() / ax2.norm();
            return uv;
        }

        WarpFunctorSimilarity(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, const double w_)
                : w(std::sqrt(w_)), gridPt1(p1), gridPt2(p2), gridPt3(p3){
            refUV = getLocalCoord<double>(p1, p2, p3);
        }

        template<typename T>
        bool operator()(const T* const g1, const T* const g2, const T* const g3, T* residual)const{
            Eigen::Matrix<T,2,1> p1(g1[0] + (T)gridPt1[0], g1[1] + (T)gridPt1[1]);
            Eigen::Matrix<T,2,1> p2(g2[0] + (T)gridPt2[0], g2[1] + (T)gridPt2[1]);
            Eigen::Matrix<T,2,1> p3(g3[0] + (T)gridPt3[0], g3[1] + (T)gridPt3[1]);
            //Eigen::Matrix<T,2,1> curUV = getLocalCoord<T>(p1,p2,p3);
            Eigen::Matrix<T,2,1> axis1 = p3-p2;
            Eigen::Matrix<T,2,1> axis2(-1.0*axis1[1], axis1[0]);
            //Matrix<T,2,1> reconp1 = axis1 * refUV[0] + axis2 * refUV[1];
            T reconx = axis1[0] * refUV[0] + axis2[0] * refUV[1] + p2[0];
            T recony = axis1[1] * refUV[0] + axis2[1] * refUV[1] + p2[1];
            T diffx = reconx - p1[0];
            T diffy = recony - p1[1];
            residual[0] = ceres::sqrt(diffx * diffx + diffy * diffy + (T)1e-5) * w;
            return true;
        }
    private:
        Eigen::Vector2d refUV;
        const double w;
        const Eigen::Vector2d& gridPt1;
        const Eigen::Vector2d& gridPt2;
        const Eigen::Vector2d& gridPt3;

    };
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_STAB_ENERGY_H
