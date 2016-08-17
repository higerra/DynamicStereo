//
// Created by yanhang on 8/16/16.
//

#ifndef DYNAMICSTEREO_COLORGMM_H
#define DYNAMICSTEREO_COLORGMM_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace dynamic_stereo {
    class ColorGMM {
    public:
        static const int componentsCount = 5;

        ColorGMM();

        float operator()(const cv::Vec3f &color) const;

        float operator()(int ci, const cv::Vec3f &color) const;

//        void computeMat(const cv::Mat& input, std::vector<cv::Mat>& output) const;
//
//        void computeProbMat(const cv::Mat& input, cv::Mat& output) const;

        int whichComponent(const cv::Vec3f& color) const;

        void initLearning();

        void addSample(int ci, const cv::Vec3f& color);

        void endLearning();

        inline float det3(const float* c) const {
            return c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                   c[2] * (c[3] * c[7] - c[4] * c[6]);
        }

        inline float det3(const Eigen::Matrix3f& m) const {
            return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
                   m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
                   m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        }

#ifdef USE_CUDA
        void computeRawProbCuda(const int ci, const cv::Mat& image, cv::Mat& output) const;
#endif
    private:
        void calcInverseCovAndDeterm(int ci);

        std::vector<float> coefs;
        std::vector<Eigen::Vector3f> mean;
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::ColMajor> > cov;
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::ColMajor> > inverseCovs;
        std::vector<float> covDeterms;

        std::vector<Eigen::Vector3f> sums;
        std::vector<Eigen::Matrix3f> prods;
        std::vector<int> sampleCounts;
        int totalSampleCount;
    };
}

#endif //DYNAMICSTEREO_COLORGMM_H
