//
// Created by yanhang on 9/11/16.
//

#include "stabilization.h"
#include "stab_energy.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {

    //structure for rectanglar grid
    struct WarpGrid{
        WarpGrid(const int width, const int height, const int gridW_, const int gridH_);
        std::vector<Eigen::Vector2d> gridLoc;
        int gridW, gridH;
        double blockW, blockH;
    };

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
                    gridLoc[y * (gridW + 1) + x][1] -= 1.1;
            }
        }
    }

    static void getGridIndAndWeight(const WarpGrid& grid, const Eigen::Vector2d& pt, Eigen::Vector4i& ind, Eigen::Vector4d& w){
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
        CHECK_GT(s, 0) << "pt: " << pt[0] << ' '<< pt[1] << " (x,y): " << x << ' ' << y;
        w = w / s;

        // if(pt.norm() <= numeric_limits<double>::min()){
        //     printf("ind: (%d,%d,%d,%d), w:(%.5f,%.5f,%.5f,%.5f)\n", ind[0], ind[1], ind[2], ind[3], w[0], w[1], w[2], w[3]);
        //     for(auto i=0; i<4; ++i)
        // 	printf("(%.2f,%.2f)\n", gridLoc[ind[i]][0], gridLoc[ind[i]][1]);
        // }
        Vector2d pt2 =
                gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
        double error = (pt2 - pt).norm();
        CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
    }

    static void warpbyGrid(const cv::Mat& input, cv::Mat& output, const WarpGrid& grid){
        output = input.clone();
        WarpGrid baseGrid(input.cols, input.rows, grid.gridW, grid.gridH);
        for(auto y=0; y<output.rows - 1; ++y){
            for(auto x=0; x<output.cols - 1; ++x){
                Vector4d biW;
                Vector4i biInd;
                getGridIndAndWeight(baseGrid, Vector2d(x,y), biInd, biW);
                Vector2d pt(0,0);
                for(auto i=0; i<4; ++i){
                    pt[0] += grid.gridLoc[biInd[i]][0] * biW[i];
                    pt[1] += grid.gridLoc[biInd[i]][1] * biW[i];
                }
                if(pt[0] < 0 || pt[1] < 0 || pt[0] > input.cols - 1 || pt[1] > input.rows - 1)
                    continue;
                Vector3d pixO = interpolation_util::bilinear<uchar,3>(input.data, input.cols, input.rows, pt);
                output.at<Vec3b>(y,x) = Vec3b((uchar)pixO[0], (uchar)pixO[1], (uchar)pixO[2]);
            }
        }
    }

    static bool isOnGrid(const WarpGrid& grid, const Eigen::Vector2d& pt, const double margin = 0.01){
        for(const auto& loc: grid.gridLoc){
            if(std::abs(pt[0] - loc[0]) < margin)
                return true;
            else if(std::abs(pt[1] - loc[1]) < margin)
                return true;
        }
        return false;
    }


    void gridStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double ws, const int step) {
        CHECK(!input.empty());
        output.resize(input.size());

        //Don't forget to copy the first frame!
        output[0] = input[0].clone();

        const int width = input[0].cols;
        const int height = input[0].rows;
        const int gridW = 5, gridH = 5;
        WarpGrid grid(width, height, gridW, gridH);

        printf("width: %d, height: %d\n", width, height);

        //aggregated warping field
        vector<vector<Vector2d> > warpingField(input.size());
        for (auto &wf: warpingField)
            wf.resize(grid.gridLoc.size(), Vector2d(0,0));

        //apply near-hard boundary constraint
        const double weight_boundary = 100;
        //variables for optimization: offset from original grid.
        //Note, defination different from old gridWarpping
        //from frame v to frame v-1

        ceres::Solver::Options ceres_option;
        ceres_option.max_num_iterations = 100;
        ceres_option.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

        //Pay attention to the directioin of warping:
        //to match frame v against v-1, we need to compute warping
        //field from v-1 to v
//#pragma omp parallel for
        for (auto v = 1; v < input.size(); ++v) {
            printf("Frame %d -> %d\n", v, v-1);
            vector<vector<double> > vars(grid.gridLoc.size());
            for (auto &var: vars) {
                var.resize(2, 0.0);
                var[0] = -4;
                var[1] = 1;
            }
            //create problem
            ceres::Problem problem;

            //appearance term
            int kDataBlock = 0;
            for (auto y = 0; y < height - 1; y += step) {
                for (auto x = 0; x < width - 1; x += step) {
                    //skip pixels on the grid to ease boundary handling
                    // if(isOnGrid(grid, Vector2d(x,y)))
                    //     continue;

                    Vec3b pix = input[v].at<Vec3b>(y, x);
                    Vector3d tgtColor((double) pix[0], (double) pix[1], (double) pix[2]);
                    Vector4i biInd;
                    Vector4d biW;

                    getGridIndAndWeight(grid, Vector2d(x, y), biInd, biW);

                    ceres::CostFunction *cost_data =
                            new ceres::NumericDiffCostFunction<WarpFunctorDense, ceres::CENTRAL, 1, 2, 2, 2, 2>(
                                    new WarpFunctorDense(input[v-1], tgtColor, biInd, biW, grid.gridLoc, 1.0)
                            );
                    problem.AddResidualBlock(cost_data, NULL, vars[biInd[0]].data(), vars[biInd[1]].data(),
                                             vars[biInd[2]].data(), vars[biInd[3]].data());

                    kDataBlock += 1;
                }
            }
            printf("Number of data block:%d\n", kDataBlock);
            //boundary constraint
//            ceres::CostFunction *cost_boundary = new ceres::AutoDiffCostFunction<WarpFunctorFix, 1, 2>(
//                    new WarpFunctorFix(nullptr, weight_boundary)
//            );
//            for (auto x = 0; x <= gridW; ++x) {
//                problem.AddResidualBlock(cost_boundary, NULL, vars[x].data());
//                problem.AddResidualBlock(cost_boundary, NULL, vars[gridH * (gridW + 1) + x].data());
//            }
//            for (auto y = 0; y <= gridH; ++y) {
//                problem.AddResidualBlock(cost_boundary, NULL, vars[y * (gridW + 1)].data());
//                problem.AddResidualBlock(cost_boundary, NULL, vars[y * (gridW + 1) + gridW].data());
//            }

            //similarity term
            for(auto y=1; y<=gridH; ++y){
                for(auto x=0; x< gridW; ++x){
                    int gid1, gid2, gid3;
                    gid1 = y * (gridW + 1) + x;
                    gid2 = (y - 1) * (gridW + 1) + x;
                    gid3 = y * (gridW + 1) + x + 1;
                    problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
                                    new WarpFunctorSimilarity(grid.gridLoc[gid1], grid.gridLoc[gid2], grid.gridLoc[gid3], ws)),
                            NULL,
                            vars[gid1].data(), vars[gid2].data(), vars[gid3].data()
                    );
                    gid2 = (y - 1) * (gridW + 1) + x + 1;
                    problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
                                    new WarpFunctorSimilarity(grid.gridLoc[gid1], grid.gridLoc[gid2], grid.gridLoc[gid3], ws)),
                            NULL,
                            vars[gid1].data(), vars[gid2].data(), vars[gid3].data());
                }
            }

            //solve
            ceres::Solver::Summary summary;
            float start_t = cv::getTickCount();
            ceres::Solve(ceres_option, &problem, &summary);
            cout << summary.BriefReport() << endl;

            for(auto i=0; i<vars.size(); ++i){
                warpingField[v][i][0] = vars[i][0];
                warpingField[v][i][1] = vars[i][1];
            }
        }

        for(auto v=1; v<input.size(); ++v){
            for(auto i=0; i<warpingField[v].size(); ++i)
                warpingField[v][i] += warpingField[v-1][i];
        }

        printf("Warping...\n");
//#pragma omp parallel for
        for(auto v=1; v<input.size(); ++v) {
            WarpGrid warpGrid(width, height, gridW, gridH);
            for (auto i = 0; i < warpingField[v].size(); ++i) {
                warpGrid.gridLoc[i] += warpingField[v][i];
            }
            warpbyGrid(input[v], output[v], warpGrid);
        }
    }
}//namespace dynamic_stereo
