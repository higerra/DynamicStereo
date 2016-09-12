//
// Created by yanhang on 9/11/16.
//

#include "stabilization.h"
#include "stab_energy.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
    WarpGrid::WarpGrid(const int width, const int height, const int gridW_, const int gridH_)
            : gridW(gridW_), gridH(gridH_) {
        blockW = (double) width / (double) gridW;
        blockH = (double) height / (double) gridH;
        gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
        for (auto x = 0; x <= gridW; ++x) {
            for (auto y = 0; y <= gridH; ++y) {
                gridLoc[y * (gridW + 1) + x] = Vector2d(blockW * x, blockH * y);
                if (x == gridW)
                    gridLoc[y * (gridW + 1) + x][0] -= 1.1;
                if (y == gridH)
                    gridLoc[y * (gridW + 1) + x][1] -= 1.0;
            }
        }
    }

    void getGridIndAndWeight(const WarpGrid& grid, const Eigen::Vector2d& pt, Eigen::Vector4i& ind, Eigen::Vector4d& w){
        const double& blockW = grid.blockW;
        const double& blockH = grid.blockH;
        const int& gridW = grid.gridW;
        const int& gridH = grid.gridH;
        const vector<Vector2d>& gridLoc = grid.gridLoc;

        int x = (int) floor(pt[0] / blockW);
        int y = (int) floor(pt[1] / blockH);
        CHECK_LE(x, gridW);
        CHECK_LE(y, gridH);

        ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
                       (y + 1) * (gridW + 1) + x);

        const double &xd = pt[0];
        const double &yd = pt[1];
        const double& xl = gridLoc[ind[0]][0];
        const double& xh = gridLoc[ind[2]][0];
        const double& yl = gridLoc[ind[0]][1];
        const double& yh = gridLoc[ind[2]][1];

        w[0] = (xh - xd) * (yh - yd);
        w[1] = (xd - xl) * (yh - yd);
        w[2] = (xd - xl) * (yd - yl);
        w[3] = (xh - xd) * (yd - yl);

        double s = w[0] + w[1] + w[2] + w[3];
        CHECK_GT(s, 0) << pt[0] << ' '<< pt[1];
        w = w / s;

        Vector2d pt2 =
                gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
        double error = (pt2 - pt).norm();
        CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
    }

    void gridStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double ws, const int step){
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;
        const int gridW = 64, gridH = 64;
        WarpGrid grid(width, height, gridW, gridH);

        //variables for optimization: offset from original grid.
        //Note, defination different from old gridWarpping
        //from frame v to frame v-1
        for(auto v=1; v<input.size(); ++v) {
            vector<vector<double> > vars(grid.gridLoc.size());
            for (auto &v: vars)
                v.resize(2, 0.0);
            //create problem
            ceres::Problem problem;
            
            //appearance term
            for(auto y=0 y < height; y+=step){
                for(auto x=0; x<width; x+=step){

                }
            }
            //boundary constraint

            //similarity term

        }
    }
}//namespace dynamic_stereo