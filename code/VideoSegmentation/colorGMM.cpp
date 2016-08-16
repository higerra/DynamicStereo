//
// Created by yanhang on 8/16/16.
//

#include "colorGMM.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    ColorGMM::ColorGMM() {
        const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
        coefs.resize(componentsCount, 0.0);
        mean.resize(componentsCount);
        cov.resize(componentsCount);
        inverseCovs.resize(componentsCount);
        covDeterms.resize(componentsCount, 0.0);

        sums.resize(componentsCount);
        prods.resize(componentsCount);
        sampleCounts.resize(componentsCount);
    }

    double ColorGMM::operator()(const Vec3d &color) const {
        double res = 0;
        for (int ci = 0; ci < componentsCount; ci++)
            res += coefs[ci] * (*this)(ci, color);
        return res;
    }

    double ColorGMM::operator()(int ci, const Vec3d &color) const {
        double res = 0;
        if (coefs[ci] > 0) {
            Vector3d diff(color[0], color[1], color[2]);
            diff -= mean[ci];
            double mult = diff.transpose() * inverseCovs[ci] * diff;
            res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
        }
        return res;
    }

    int ColorGMM::whichComponent(const Vec3d& color) const {
        int k = 0;
        double max = 0;
        for (int ci = 0; ci < componentsCount; ci++) {
            double p = (*this)(ci, color);
            if (p > max) {
                k = ci;
                max = p;
            }
        }
        return k;
    }

    void ColorGMM::initLearning() {
        for (int ci = 0; ci < componentsCount; ci++) {
            sums[ci] = Vector3d::Zero();
            prods[ci] = Matrix3d::Zero();
            sampleCounts[ci] = 0;
        }
        totalSampleCount = 0;
    }

    void ColorGMM::addSample(int ci, const Vec3d& color) {
        Eigen::Map<Eigen::Vector3d> data(const_cast<double*>(&color[0]));
        sums[ci] += data;
        prods[ci] += data * data.transpose();
        sampleCounts[ci]++;
        totalSampleCount++;
    }

    void ColorGMM::endLearning() {
        const double variance = 0.01;
        for (int ci = 0; ci < componentsCount; ci++) {
            int n = sampleCounts[ci];
            if (n == 0)
                coefs[ci] = 0;
            else {
                coefs[ci] = (double) n / totalSampleCount;

                mean[ci] = sums[ci] / n;
                cov[ci] = prods[ci] / n - mean[ci] * mean[ci].transpose();

                double dtrm = det3(cov[ci]);
                if (dtrm <= std::numeric_limits<double>::epsilon()) {
                    // Adds the white noise to avoid singular covariance matrix.
                    cov[ci](0,0) += variance;
                    cov[ci](1,1) += variance;
                    cov[ci](2,2) += variance;
                }

                calcInverseCovAndDeterm(ci);
            }
        }
    }

    void ColorGMM::calcInverseCovAndDeterm(int ci) {
        if (coefs[ci] > 0) {
            covDeterms[ci] = det3(cov[ci]);
            inverseCovs[ci] = cov[ci].inverse();
        }
    }

//    void ColorGMM::computeMat(const cv::Mat& input, std::vector<cv::Mat>& output) const{
//        CHECK(!input.empty());
//        const int width = input.cols;
//        const int height = input.rows;
//        Mat rawdata;
//        input.convertTo(rawdata, CV_64FC3);
//
//        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, 3>;
//
//        output.resize(componentsCount);
//        //map input data to eigen matrix
//        Eigen::Map<MatrixType> samples((double*)rawdata.data, width * height, 3);
//#pragma omp parallel for
//        for(auto ci=0; ci < componentsCount; ++ci) {
//            output[ci].create(height, width, CV_64FC1);
//            double coe = 1.0 / sqrt(covDeterms[ci]);
//
//            MatrixType m = mean[ci].transpose().replicate(width * height, 1);
//            MatrixType diff = samples - m;
//            MatrixType mid = diff * inverseCovs[ci];
//
//            for(auto y=0; y<height; ++y){
//                for(auto x=0; x<width; ++x)
//                    output[ci].at<double>(y,x) = coe * std::exp(-0.5 * mid.row(y*width+x).dot(diff.row(y*width + x)));
//            }
//        }
//    }
//
//    void ColorGMM::computeProbMat(const cv::Mat& input, cv::Mat& output) const{
//        vector<Mat> rawProb;
//        computeMat(input, rawProb);
//        output.create(input.size(), CV_64FC1);
//        output.setTo(0.0);
//
//        double coeSum = std::accumulate(coefs.begin(), coefs.end(), 0.0);
//        for(auto ci=0; ci<componentsCount; ++ci)
//            output += rawProb[ci] * coefs[ci];
//        output /= coeSum;
//    }
}//namespace dynamic_stereo