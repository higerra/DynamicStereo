//
// Created by yanhang on 4/21/16.
//

#ifndef SUBSPACESTAB_WARPPING_H
#define SUBSPACESTAB_WARPPING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <glog/logging.h>

namespace substab {

    //experimental: solve inverse bilinear
    // axy + bx + cy + d = 0
    // exy + fx + gy + h = 0
    // only return real solution, otherwise return (-1,-1)
    bool solveInverseBilinear(const double a_, const double b_, const double c_, const double d_,
                              const double e_, const double f_, const double g_, const double h_,
                              std::vector<Eigen::Vector2d> & res);

    class GridWarpping {
    public:
        GridWarpping(const int w, const int h, const int gw = 64, const int gh = 36);

        inline int gridInd(int x, int y)const{
            CHECK_LE(x, gridW);
            CHECK_LE(y, gridH);
            return y*(gridW+1)+x;
        }
        inline double getBlockW() const{
            return blockW;
        }
        inline double getBlockH() const{
            return blockH;
        }

        inline Eigen::Vector2d warpPoint(const Eigen::Vector2d& pt) const{
            Eigen::Vector4i ind;
            Eigen::Vector4d w;
            getGridIndAndWeight(pt, ind, w);
            Eigen::Vector2d res = gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
            return res;
        }
        bool inverseWarpPoint(const Eigen::Vector2d& pt, Eigen::Vector2d& res) const;

        inline const std::vector<Eigen::Vector2d>& getBaseGrid() const{return gridLoc;}
        inline const std::vector<Eigen::Vector2d>& getWarpedGrid() const{return warpedLoc;}
        inline std::vector<Eigen::Vector2d>& getWarpedGrid() {return warpedLoc;}


        void getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind, Eigen::Vector4d &w) const;

        void computeSimilarityWeight(const cv::Mat& input, std::vector<double>& saliency) const;

        //compute warping field from src to tgt. (warp src to match tgt)
        void computeWarpingField(const std::vector<Eigen::Vector2d>& src, const std::vector<Eigen::Vector2d>& tgt,
                                 const bool fixBoundary = false);

        //forward warp: for each pixel of input, new location is computed based on warping field and the color is gaussian splatted
        //backward warp: for each pixel of output, compute the location in the input based on warping field
        //For example, if you want to register f2 to f1, you can either:
        //     1. compute warping field from f1->f2 and do backward warping
        //     2. compute warping field from f2->f1 and do forward warping
        //Use forward warping only when necessary, since the gaussian splatting may introduce blury.
        void warpImageForward(const cv::Mat& input, cv::Mat& output, const double splattR = 1) const;
        void warpImageBackward(const cv::Mat& input, cv::Mat& output) const;

        void visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat& img) const;
    private:
        std::vector<Eigen::Vector2d> gridLoc;
        std::vector<Eigen::Vector2d> warpedLoc;

        int width;
        int height;
        int gridW;
        int gridH;
        double blockW;
        double blockH;
    };
}//namespace substab

#endif //SUBSPACESTAB_WARPPING_H
