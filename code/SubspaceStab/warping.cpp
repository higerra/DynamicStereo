//
// Created by yanhang on 4/21/16.
//

#include "warping.h"
//#include "gridenergy.h"
#include "../base/utility.h"
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace substab{

    bool solveInverseBilinear(const double a_, const double b_, const double c_, const double d_,
                              const double e_, const double f_, const double g_, const double h_,
                              std::vector<Eigen::Vector2d>& res) {
        bool swapXY = false;
        double a = a_, b = b_, c = c_, d = d_, e = e_, f = f_, g = g_, h = h_;
        if (a * g - e * c < std::numeric_limits<double>::epsilon()) {
            std::swap(b, c);
            std::swap(f, g);
            swapXY = true;
        }
        if (std::abs(a * g - e * c) < std::numeric_limits<double>::epsilon()) {
            LOG(WARNING) << "Ill condition: ag - ec = 0";
            return false;
        }
        const double div = a * g - e * c;

        //quadratic problem
        const double A = (a * e * b - a * a * f) / div;
        const double B = (a * e * d - a * a * h) / div + b + (e * c * b - a * c * f) / div;
        const double C = (e * c * d - a * c * h) / div + d;

        if(std::abs(A) < std::numeric_limits<double>::epsilon()){
            LOG(WARNING) << "Ill condition: A = 0";
            return false;
        }
        const double J = B * B - 4 * A * C;
        if(J < 0){
            LOG(WARNING) << "No real solution " << A << ' ' << B << ' ' << C << ' ' << J;
            return false;
        }

        if(J <= std::numeric_limits<double>::epsilon()){
            Vector2d r1;
            r1[0] = -B / (2 * A);
            r1[1] = (e*b-a*f) / div * r1[0] + (e*d-a*h) / div;
            if(swapXY)
                std::swap(r1[0], r1[1]);
            res.push_back(r1);
        }else{
            Vector2d r1, r2;
            r1[0] = (-1*B + std::sqrt(J)) / (2*A);
            r2[0] = (-1*B - std::sqrt(J)) / (2*A);

            r1[1] = (e*b-a*f) / div * r1[0] + (e*d-a*h) / div;
            r2[1] = (e*b-a*f) / div * r2[0] + (e*d-a*h) / div;

            if(swapXY){
                std::swap(r1[0], r1[1]);
                std::swap(r2[0], r2[1]);
                res.push_back(r1);
                res.push_back(r2);
            }
        }
        return true;
    }

    GridWarpping::GridWarpping(const int w, const int h, const int gw, const int gh) : width(w), height(h), gridW(gw), gridH(gh) {
        blockW = (double) width / gridW;
        blockH = (double) height / gridH;
        gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
        for (auto x = 0; x <= gridW; ++x) {
            for (auto y = 0; y <= gridH; ++y) {
                gridLoc[y * (gridW + 1) + x] = Eigen::Vector2d(blockW * x, blockH * y);
                if (x == gridW)
                    gridLoc[y * (gridW + 1) + x][0] -= 1.1;
                if (y == gridH)
                    gridLoc[y * (gridW + 1) + x][1] -= 1.1;
            }
        }
        warpedLoc = gridLoc;

    }

    bool GridWarpping::inverseWarpPoint(const Eigen::Vector2d& pt, Eigen::Vector2d& res) const {
        vector<Vector2d> solutions;
        for (auto gx = 0; gx < gridW; ++gx) {
            for (auto gy = 0; gy < gridH; ++gy) {
                const double gxl = gridLoc[gridInd(gx, gy)][0];
                const double gxh = gridLoc[gridInd(gx + 1, gy + 1)][0];
                const double gyl = gridLoc[gridInd(gx, gy)][1];
                const double gyh = gridLoc[gridInd(gx + 1, gy + 1)][1];

                Vector4d wptx(warpedLoc[gridInd(gx, gy)][0], warpedLoc[gridInd(gx + 1, gy)][0],
                              warpedLoc[gridInd(gx + 1, gy + 1)][0], warpedLoc[gridInd(gx, gy + 1)][0]);
                Vector4d wpty(warpedLoc[gridInd(gx, gy)][1], warpedLoc[gridInd(gx + 1, gy)][1],
                              warpedLoc[gridInd(gx + 1, gy + 1)][1], warpedLoc[gridInd(gx, gy + 1)][1]);

                const double a = Vector4d(1, -1, 1, -1).dot(wptx);
                const double e = Vector4d(1, -1, 1, -1).dot(wpty);

                const double b = Vector4d(-gyh, gyh, -gyl, gyl).dot(wptx);
                const double c = Vector4d(-gxh, gxl, -gxl, gxh).dot(wptx);

                const double f = Vector4d(-gyh, gyh, -gyl, gyl).dot(wpty);
                const double g = Vector4d(-gxh, gxl, -gxl, gxh).dot(wpty);

                const double d = Vector4d(gxh * gyh, -gxl * gyh, gxl * gyl, -gxh * gyl).dot(wptx) - pt[0];
                const double h = Vector4d(gxh * gyh, -gxl * gyh, gxl * gyl, -gxh * gyl).dot(wpty) - pt[1];

                vector<Vector2d> curSol;
                if (solveInverseBilinear(a, b, c, d, e, f, g, h, curSol)){
                    for(const auto& cs: curSol){
                        if(cs[0] >= gxl && cs[0] < gxh && cs[1] >= gyl && cs[1] < gyh)
                            solutions.push_back(cs);
                    }
                }
            }
        }
        if(solutions.size() >= 1) {
            res = solutions.back();
            return true;
        }
        return false;
    }

    void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
                                           Eigen::Vector4d &w) const {

        int x = (int) floor(pt[0] / blockW);
        int y = (int) floor(pt[1] / blockH);
        if(pt[0] <= 0)
            x = 0;
        if(pt[0] >= width - 1)
            x = gridW - 1;
        if(pt[1] <= 0)
            y = 0;
        if(pt[1] >= height - 1)
            y = gridH - 1;

        //////////////
        // 1--2
        // |  |
        // 4--3
        /////////////
        ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
                       (y + 1) * (gridW + 1) + x);

        const double &xd = pt[0];
        const double &yd = pt[1];
        const double xl = gridLoc[ind[0]][0];
        const double xh = gridLoc[ind[2]][0];
        const double yl = gridLoc[ind[0]][1];
        const double yh = gridLoc[ind[2]][1];

        w[0] = (xh - xd) * (yh - yd);
        w[1] = (xd - xl) * (yh - yd);
        w[2] = (xd - xl) * (yd - yl);
        w[3] = (xh - xd) * (yd - yl);

        double s = w[0] + w[1] + w[2] + w[3];
        CHECK_GT(std::fabs(s), 0) << pt[0] << ' '<< pt[1];
        w = w / s;

        Vector2d pt2 =
                gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
        double error = (pt2 - pt).norm();
        CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
    }


    void GridWarpping::visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat &img) const {
        CHECK_EQ(grid.size(), gridLoc.size());
        CHECK_EQ(img.cols, width);
        CHECK_EQ(img.rows, height);
        for(auto gy=0; gy<gridH; ++gy) {
            for (auto gx = 0; gx < gridW; ++gx){
                const int gid1 = gy * (gridW+1) + gx;
                const int gid2 = (gy+1) * (gridW+1) + gx;
                const int gid3 = (gy+1)*(gridW+1)+gx+1;
                const int gid4= gy * (gridW+1) + gx+1;
                cv::line(img, cv::Point(grid[gid1][0], grid[gid1][1]), cv::Point(grid[gid2][0], grid[gid2][1]), Scalar(255,255,255));
                cv::line(img, cv::Point(grid[gid2][0], grid[gid2][1]), cv::Point(grid[gid3][0], grid[gid3][1]), Scalar(255,255,255));
                cv::line(img, cv::Point(grid[gid3][0], grid[gid3][1]), cv::Point(grid[gid4][0], grid[gid4][1]), Scalar(255,255,255));
                cv::line(img, cv::Point(grid[gid4][0], grid[gid4][1]), cv::Point(grid[gid1][0], grid[gid1][1]), Scalar(255,255,255));
            }
        }
    }

    void GridWarpping::computeSimilarityWeight(const cv::Mat &input, std::vector<double>& saliency) const {
        saliency.resize((size_t)(gridW*gridH));
        for(auto y=0; y<gridH; ++y){
            for(auto x=0; x<gridW; ++x){
                const int gid = gridInd(x,y);
                vector<vector<double> > pixs(3);
                for(auto x1=(int)gridLoc[gridInd(x,y)][0]; x1<gridLoc[gridInd(x+1,y+1)][0]; ++x1){
                    for(auto y1=(int)gridLoc[gridInd(x,y)][1]; y1<gridLoc[gridInd(x+1,y+1)][1]; ++y1){
                        Vec3b pix = input.at<Vec3b>(y1,x1);
                        pixs[0].push_back((double)pix[0] / 255.0);
                        pixs[1].push_back((double)pix[1] / 255.0);
                        pixs[2].push_back((double)pix[2] / 255.0);
                    }
                }
                Vector3d vars(math_util::variance(pixs[0]),math_util::variance(pixs[1]),math_util::variance(pixs[2]));
                saliency[gid] = vars.norm();
            }
        }
    }

    void GridWarpping::computeWarpingField(const std::vector<Eigen::Vector2d>& src, const std::vector<Eigen::Vector2d>& tgt,
                                           const double wsimilarity,
                                           const bool fixBoundary){
        CHECK_EQ(src.size(), tgt.size());

        const int kDataTerm = (int)src.size() * 2;

        //kSimTerm is a conservative estimation
        const int kSimTerm = (gridW)*(gridH)*8;
        const int kVar = (int)gridLoc.size() * 2;

        vector<Eigen::Triplet<double> > triplets;
        VectorXd B(kDataTerm+kSimTerm);

        //cInd: counter for constraint
        int cInd = 0;
        //add data constraint

        const double wdata = 1.0;
        for(auto i=0; i<src.size(); ++i) {
            if (src[i][0] < 0 || src[i][1] < 0 || src[i][0] >= width - 1 || src[i][1] >= height - 1)
                continue;
            Vector4i indRef;
            Vector4d bwRef;
            CHECK_LT(cInd + 1, B.rows());

            getGridIndAndWeight(src[i], indRef, bwRef);
            for (auto j = 0; j < 4; ++j) {
                CHECK_LT(indRef[j]*2+1, kVar);
                triplets.push_back(Triplet<double>(cInd, indRef[j] * 2, wdata * bwRef[j]));
                triplets.push_back(Triplet<double>(cInd + 1, indRef[j] * 2 + 1, wdata * bwRef[j]));
            }
            B[cInd] = wdata * tgt[i][0];
            B[cInd + 1] = wdata * tgt[i][1];
            cInd += 2;
        }

        if(fixBoundary){
            for(auto x=0; x<=gridW; ++x){

            }
        }
//		vector<double> saliency;
//		computeSimilarityWeight(input, saliency);

        auto getLocalCoord = [](const Vector2d& p1, const Vector2d& p2, const Vector2d& p3){
            Vector2d axis1 = p3 - p2;
            Vector2d axis2(-1*axis1[1], axis1[0]);
            Vector2d v = p1 - p2;
            return Vector2d(v.dot(axis1)/axis1.squaredNorm(), v.dot(axis2)/axis2.squaredNorm());
        };
        //clockwise
        for(auto y=0; y<= gridH; ++y) {
            for (auto x = 0; x <= gridW; ++x) {
                vector<Vector2i> gids(4, Vector2i(-1,-1));
                if(x > 0 && y > 0)
                    gids[0] = Vector2i(gridInd(x - 1, y), gridInd(x, y - 1));
                if(x < gridW && y > 0)
                    gids[1] = Vector2i(gridInd(x, y - 1), gridInd(x + 1, y));
                if(x < gridW && y < gridH)
                    gids[2] = Vector2i(gridInd(x + 1, y), gridInd(x, y + 1));
                if(x > 0 && y < gridH)
                    gids[3] = Vector2i(gridInd(x, y + 1), gridInd(x - 1, y));

                const int cgid = gridInd(x, y);
                for (const auto &gid: gids) {
                    if(gid[0] >= 0 && gid[1] >= 0){
                        Vector2d refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]);
                        //x coordinate
                        triplets.push_back(Triplet<double>(cInd, cgid * 2, wsimilarity));
                        triplets.push_back(Triplet<double>(cInd, gid[0]*2, -1 * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd, gid[0] * 2, refUV[0] * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd, gid[1] * 2, -1 * refUV[0] * wsimilarity));

                        triplets.push_back(Triplet<double>(cInd, gid[0] * 2 + 1, -1 * refUV[1] * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd, gid[1] * 2 + 1, refUV[1] * wsimilarity));
                        B[cInd] = 0;

                        //y coordinate
                        triplets.push_back(Triplet<double>(cInd + 1, cgid * 2 + 1, wsimilarity));
                        triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, -1 * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, refUV[0] * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2 + 1, -1 * refUV[0] * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2, refUV[1] * wsimilarity));
                        triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2, -1 * refUV[1] * wsimilarity));
                        B[cInd + 1] = 0;
                        cInd += 2;
                    }
                }
            }
        }

        CHECK_LE(cInd, kDataTerm+kSimTerm);
        SparseMatrix<double> A(cInd, kVar);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::SPQR<SparseMatrix<double> > solver(A);
        VectorXd res = solver.solve(B.block(0,0,cInd,1));
        CHECK_EQ(res.rows(), kVar);

        //save the optimized result
        for(auto i=0; i<warpedLoc.size(); ++i){
            warpedLoc[i][0] = res[2*i];
            warpedLoc[i][1] = res[2*i+1];
        }
    }

    void GridWarpping::warpImageForward(const cv::Mat &input, cv::Mat &output, const double splattR) const {
        CHECK_EQ(input.cols, width);
        CHECK_EQ(input.rows, height);

        Mat accPix(input.size(), CV_32FC3, Scalar::all(0));
        Mat accWeight(input.size(), CV_32FC1, Scalar::all(0));
        double sigma = 1.0;
        if(splattR >= 0.5)
            sigma = splattR / 2;

        vector<float> gauKernel;
        for (double dx = -1 * splattR; dx <= splattR; dx += 1.0) {
            for (double dy = -1 * splattR; dy <= splattR; dy += 1.0) {
                float w = std::exp(-1 * dx * dx / 2 / sigma - dy * dy / 2 / sigma);
                gauKernel.push_back(w);
            }
        }

        const double step = 1.0;
        for(double y=0; y<=height - 1; y += step) {
            for (double x = 0; x <= width - 1; x += step) {
                Vec3f pix = (Vec3f) input.at<Vec3b>(y, x);
                Vector2d basePt = warpPoint(Vector2d(x, y));
                int kIndex = 0;
                for (double dx = -1 * splattR; dx <= splattR + numeric_limits<double>::epsilon(); dx += 1.0) {
                    for (double dy = -1 * splattR; dy <= splattR + numeric_limits<double>::epsilon(); dy += 1.0, ++kIndex) {
                        Vector2d pt = basePt + Vector2d(dx, dy);
                        int ix = std::round(pt[0]);
                        int iy = std::round(pt[1]);
                        if (ix >= 0 && iy >= 0 && ix <= width - 1 && iy <= height - 1) {
                            accPix.at<Vec3f>(iy, ix) += pix * gauKernel[kIndex];
                            accWeight.at<float>(iy, ix) += gauKernel[kIndex];
                        }
                    }
                }
            }
        }

        output.create(input.size(), CV_8UC3);
        output.setTo(Scalar::all(0));

        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){
                if(accWeight.at<float>(y,x) > FLT_EPSILON){
                    Vec3f pix = accPix.at<Vec3f>(y,x) / accWeight.at<float>(y,x);
                    output.at<Vec3b>(y,x) = (Vec3b)pix;
                }
            }
        }
    }

    void GridWarpping::warpImageBackward(const cv::Mat &input, cv::Mat &output) const {
        CHECK_EQ(input.cols, width);
        CHECK_EQ(input.rows, height);
        output = Mat(height, width, CV_8UC3, Scalar::all(0));
        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){
                Vector2d pt = warpPoint(Vector2d(x,y));
                if(pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
                    continue;
                Vector3d pixO = interpolation_util::bilinear<uchar,3>(input.data, input.cols, input.rows, pt);
                output.at<Vec3b>(y,x) = Vec3b((uchar) pixO[0], (uchar)pixO[1], (uchar)pixO[2]);
            }
        }
    }

}//namespace substablas
