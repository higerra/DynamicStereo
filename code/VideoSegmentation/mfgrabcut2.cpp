//
// Created by yanhang on 6/4/16.
//

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <limits>
#include <memory>
#include "../external/MRF2.2/GCoptimization.h"
#include "videosegmentation.h"

using namespace cv;
using namespace std;
/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
namespace dynamic_stereo {
    namespace video_segment {
        class GMM {
        public:
            static const int componentsCount = 5;

            GMM();

            inline Mat& getModel() {return model;}
            inline const Mat& getModel() const{return model;}

            double operator()(const Vec3d &color) const;

            double operator()(int ci, const Vec3d &color) const;

            int whichComponent(const Vec3d color) const;

            void initLearning();

            void addSample(int ci, const Vec3d& color);

            void endLearning();

        private:
            void calcInverseCovAndDeterm(int ci);

            Mat model;
            double *coefs;
            double *mean;
            double *cov;

            double inverseCovs[componentsCount][3][3];
            double covDeterms[componentsCount];

            double sums[componentsCount][3];
            double prods[componentsCount][3][3];
            int sampleCounts[componentsCount];
            int totalSampleCount;
        };

        GMM::GMM() {
            const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
            model.create(1, modelSize * componentsCount, CV_64FC1);
            model.setTo(Scalar(0));

            coefs = model.ptr<double>(0);
            mean = coefs + componentsCount;
            cov = mean + 3 * componentsCount;

            for (int ci = 0; ci < componentsCount; ci++)
                if (coefs[ci] > 0)
                    calcInverseCovAndDeterm(ci);
        }

        double GMM::operator()(const Vec3d &color) const {
            double res = 0;
            for (int ci = 0; ci < componentsCount; ci++)
                res += coefs[ci] * (*this)(ci, color);
            return res;
        }

        double GMM::operator()(int ci, const Vec3d &color) const {
            double res = 0;
            if (coefs[ci] > 0) {
                CHECK_GT(covDeterms[ci], std::numeric_limits<double>::epsilon());
                Vec3d diff = color;
                double *m = mean + 3 * ci;
                diff[0] -= m[0];
                diff[1] -= m[1];
                diff[2] -= m[2];
                double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] +
                                         diff[2] * inverseCovs[ci][2][0])
                              + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] +
                                           diff[2] * inverseCovs[ci][2][1])
                              + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] +
                                           diff[2] * inverseCovs[ci][2][2]);
                res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
            }
            return res;
        }

        int GMM::whichComponent(const Vec3d color) const {
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

        void GMM::initLearning() {
            for (int ci = 0; ci < componentsCount; ci++) {
                sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
                prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
                prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
                prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
                sampleCounts[ci] = 0;
            }
            totalSampleCount = 0;
        }

        void GMM::addSample(int ci, const Vec3d& color) {
            sums[ci][0] += color[0];
            sums[ci][1] += color[1];
            sums[ci][2] += color[2];
            prods[ci][0][0] += color[0] * color[0];
            prods[ci][0][1] += color[0] * color[1];
            prods[ci][0][2] += color[0] * color[2];
            prods[ci][1][0] += color[1] * color[0];
            prods[ci][1][1] += color[1] * color[1];
            prods[ci][1][2] += color[1] * color[2];
            prods[ci][2][0] += color[2] * color[0];
            prods[ci][2][1] += color[2] * color[1];
            prods[ci][2][2] += color[2] * color[2];
            sampleCounts[ci]++;
            totalSampleCount++;
        }

        void GMM::endLearning() {
            const double variance = 0.01;
            for (int ci = 0; ci < componentsCount; ci++) {
                int n = sampleCounts[ci];
                if (n == 0)
                    coefs[ci] = 0;
                else {
                    coefs[ci] = (double) n / totalSampleCount;

                    double *m = mean + 3 * ci;
                    m[0] = sums[ci][0] / n;
                    m[1] = sums[ci][1] / n;
                    m[2] = sums[ci][2] / n;

                    double *c = cov + 9 * ci;
                    c[0] = prods[ci][0][0] / n - m[0] * m[0];
                    c[1] = prods[ci][0][1] / n - m[0] * m[1];
                    c[2] = prods[ci][0][2] / n - m[0] * m[2];
                    c[3] = prods[ci][1][0] / n - m[1] * m[0];
                    c[4] = prods[ci][1][1] / n - m[1] * m[1];
                    c[5] = prods[ci][1][2] / n - m[1] * m[2];
                    c[6] = prods[ci][2][0] / n - m[2] * m[0];
                    c[7] = prods[ci][2][1] / n - m[2] * m[1];
                    c[8] = prods[ci][2][2] / n - m[2] * m[2];

                    double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                                  c[2] * (c[3] * c[7] - c[4] * c[6]);
                    if (dtrm <= std::numeric_limits<double>::epsilon()) {
                        // Adds the white noise to avoid singular covariance matrix.
                        c[0] += variance;
                        c[4] += variance;
                        c[8] += variance;
                    }

                    calcInverseCovAndDeterm(ci);
                }
            }
        }

        void GMM::calcInverseCovAndDeterm(int ci) {
            if (coefs[ci] > 0) {
                double *c = cov + 9 * ci;
                double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                              c[2] * (c[3] * c[7] - c[4] * c[6]);
                covDeterms[ci] = dtrm;

                CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
                inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
                inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
                inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
                inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
                inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
                inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
                inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
                inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
                inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
            }
        }

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
        static double calcBeta(const vector<Mat> &images) {
            double beta = 0;
            double count = 0.0;
            for (const auto &img: images) {
                for (int y = 0; y < img.rows; y++) {
                    for (int x = 0; x < img.cols; x++) {
                        Vec3d color = img.at<Vec3b>(y, x);
                        if (x > 0) // left
                        {
                            Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                            beta += diff.dot(diff);
                            count += 1.0;
                        }
                        if (y > 0 && x > 0) // upleft
                        {
                            Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                            beta += diff.dot(diff);
                            count += 1.0;
                        }
                        if (y > 0) // up
                        {
                            Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                            beta += diff.dot(diff);
                            count += 1.0;
                        }
                        if (y > 0 && x < img.cols - 1) // upright
                        {
                            Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                            beta += diff.dot(diff);
                            count += 1.0;
                        }
                    }
                }
            }
            if (beta <= std::numeric_limits<double>::epsilon())
                beta = 0;
            else
                beta = 1.f / (2 * beta / count);

            return beta;
        }

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
        static void calcNWeights(const Mat &img, Mat &leftW, Mat &upleftW, Mat &upW, Mat &uprightW, double beta,
                                 double gamma) {
            const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
            leftW.create(img.rows, img.cols, CV_64FC1);
            upleftW.create(img.rows, img.cols, CV_64FC1);
            upW.create(img.rows, img.cols, CV_64FC1);
            uprightW.create(img.rows, img.cols, CV_64FC1);
            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    Vec3d color = img.at<Vec3b>(y, x);
                    if (x - 1 >= 0) // left
                    {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                        leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                    } else
                        leftW.at<double>(y, x) = 0;
                    if (x - 1 >= 0 && y - 1 >= 0) // upleft
                    {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                        upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                    } else
                        upleftW.at<double>(y, x) = 0;
                    if (y - 1 >= 0) // up
                    {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                        upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                    } else
                        upW.at<double>(y, x) = 0;
                    if (x + 1 < img.cols && y - 1 >= 0) // upright
                    {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                        uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                    } else
                        uprightW.at<double>(y, x) = 0;
                }
            }
        }

        /*
      Check size, type and element values of mask matrix.
     */
        static int checkMask(const Mat &img, const Mat &mask) {
            CHECK(!mask.empty()) << "Mask is empty";
            CHECK_EQ(mask.type(), CV_32SC1);
            CHECK_EQ(mask.size(), img.size());

            //check whether the elements is contiguous
            double minL, maxL;
            minMaxLoc(mask, &minL, &maxL);
            const int nLabel = (int)maxL + 1;
            std::vector<bool> occupied ((size_t)nLabel, false);
            for (int y = 0; y < mask.rows; y++) {
                for (int x = 0; x < mask.cols; x++) {
                    int val = mask.at<int>(y, x);
                    CHECK_LT(val, nLabel);
                    occupied[val] = true;
                }
            }

            for(auto i=0; i<nLabel; ++i)
                CHECK(occupied[i]) << "Label " << i << " is missing.";
            return nLabel;
        }

        //preprocessing: for each label, only modify center part.
        static int preprocessMask(const Mat& img, Mat& mask, Mat& hardConstraint){
            const int nLabel = checkMask(img, mask);

            //only modify $shrinkRatio on the border
            const double shrinkRatio = 0.02;
            const double minShrink = 3;

            hardConstraint.create(mask.size(), CV_8UC1);
            hardConstraint.setTo(Scalar::all(0));

            int *pMask = (int *) mask.data;
            vector<Mat> labelMask((size_t)nLabel);
            vector<vector<cv::Point> > labelPoints((size_t)nLabel);
            for(auto& m: labelMask) {
                m.create(mask.size(), CV_8UC1);
                m.setTo(Scalar::all(0));
            }
            for(auto i=0; i<mask.cols * mask.rows; ++i){
                labelMask[pMask[i]].data[i] = (uchar)255;
                labelPoints[pMask[i]].emplace_back(i/mask.cols, i%mask.cols);
            }


            for(auto lid=0; lid < labelMask.size(); ++lid){
                cv::RotatedRect bbox = cv::minAreaRect(labelPoints[lid]);
                int r = std::max(std::min(bbox.size.height, bbox.size.width) * shrinkRatio, minShrink);
                Mat structure = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(2*r+1, 2*r+1));
                cv::erode(labelMask[lid], labelMask[lid], structure);
                for(auto i=0; i<mask.cols * mask.rows; ++i){
                    if(labelMask[lid].data[i] > (uchar)200)
                        hardConstraint.data[i] = (uchar)255;
                }
            }

            return nLabel;
        }

        /*
      Initialize GMM background and foreground models using kmeans algorithm.
    */
        static void initGMMs(const vector<Mat> &images, const Mat &mask, vector<GMM>& gmms) {
            CHECK(!gmms.empty());
            const int kMeansItCount = 10;
            const int kMeansType = KMEANS_PP_CENTERS;
            const int nLabel = gmms.size();

            std::vector<std::vector<Vec3f> > samples(gmms.size());
            Point p;
            for (const auto &img: images) {
                for (p.y = 0; p.y < img.rows; p.y++) {
                    for (p.x = 0; p.x < img.cols; p.x++) {
                        const int comId = mask.at<int>(p);
                        samples[comId].push_back((Vec3f) img.at<Vec3b>(p));
                    }
                }
            }
            for(const auto& s: samples)
                CHECK(!s.empty());
#pragma omp parallel for
            for(auto l=0; l<samples.size(); ++l){
                Mat bestLabels;
                Mat sampleMat((int) samples[l].size(), 3, CV_32FC1, &samples[l][0][0]);
                kmeans(sampleMat, GMM::componentsCount, bestLabels, TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

                gmms[l].initLearning();
                for(auto i=0; i<samples[l].size(); ++i)
                    gmms[l].addSample(bestLabels.at<int>(i,0), samples[l][i]);
                gmms[l].endLearning();
            }
        }
/*
  Assign GMMs components for each pixel.
*/
        static void assignGMMsComponents(const Mat &img, const Mat &mask, const std::vector<GMM>& gmms,
                                         Mat &compIdxs) {
            Point p;
            for (p.y = 0; p.y < img.rows; p.y++) {
                for (p.x = 0; p.x < img.cols; p.x++) {
                    Vec3d color = (Vec3d)img.at<Vec3b>(p);
                    const int lid = mask.at<int>(p);
                    compIdxs.at<int>(p) = gmms[lid].whichComponent(color);
                }
            }
        }

/*
  Learn GMMs parameters.
*/
        static void learnGMMs(const vector<Mat> &images, const Mat &mask, const vector<Mat> &compIdxs_all, vector<GMM>& gmms) {
            for(auto& g: gmms)
                g.initLearning();
            for(auto v=0; v<images.size(); ++v){
                const Mat &img = images[v];
                const Mat &compIdxs = compIdxs_all[v];
                for (auto y = 0; y < img.rows; ++y) {
                    for (auto x = 0; x < img.cols; ++x) {
                        const int comId = compIdxs.at<int>(y,x);
                        const int lid = mask.at<int>(y,x);
                        gmms[lid].addSample(comId, (Vec3d) img.at<Vec3b>(y,x));
                    }
                }
            }
            for(auto& g: gmms)
                g.endLearning();
        }

/*
  Construct GCGraph
*/
        static void runGraphCut(const vector<Mat> &images, Mat &mask, const Mat& hardConstraint,
                                const std::vector<GMM>& gmms,
                                double lambda,
                                const vector<Mat> &leftWs, const vector<Mat> &upleftWs, const vector<Mat> &upWs,
                                const vector<Mat> &uprightWs){
            CHECK(!images.empty());
            const int width = images[0].cols;
            const int height = images[0].rows;
            const int nLabel = (int)gmms.size();
            int vtxCount = width * height;
            Point p;

            vector<double> MRF_data((size_t) nLabel * vtxCount, 0.0);
            vector<double> MRF_smoothess((size_t) nLabel * nLabel, 1.0);
            for(auto l1=0; l1 < nLabel; ++l1){
                for(auto l2=0; l2 < nLabel; ++l2){
                    if(l1 == l2)
                        MRF_smoothess[l1*nLabel+l2] = 0.0;
                }
            }

#pragma omp parallel for
            for(auto y=0; y < height; ++y){
                for(auto x = 0; x < width; ++x){
                    const int vtxIdx = y * width + x;
                    //assign data term
                    for(auto l=0; l<nLabel; ++l){
                        double e = 0.0;
                        if(hardConstraint.at<uchar>(y,x) > (uchar)200){
                            if(l != mask.at<int>(y,x))
                                e = lambda * (double)images.size();
                        }else {
                            for (const auto &img: images) {
                                Vec3d color = (Vec3d) img.at<Vec3b>(y,x);
                                e -= log(gmms[l](color));
                            }
                        }
                        MRF_data[vtxIdx * nLabel + l] = e;
                    }
                }
            }

            std::shared_ptr<DataCost> dataCost(
                    new DataCost(MRF_data.data())
            );
            std::shared_ptr<SmoothnessCost> smoothnessCost(
                    new SmoothnessCost(MRF_smoothess.data())
            );

            std::shared_ptr<EnergyFunction> energy_function(
                    new EnergyFunction(dataCost.get(), smoothnessCost.get())
            );

            std::shared_ptr<Expansion> mrf(
                    new Expansion(vtxCount, nLabel, energy_function.get())
            );

            mrf->initialize();

            for(p.y = 0; p.y < height; ++p.y){
                for(p.x=0; p.x < width; ++p.x){
                    //assign smoothness weight
                    const int vtxIdx = p.y * width + p.x;
                    if (p.x > 0) {
                        double w = 0.0;
                        for (const auto &leftW: leftWs) {
                            w += leftW.at<double>(p);
                        }
                        mrf->setNeighbors(vtxIdx, vtxIdx-1, w);
                    }
                    if (p.x > 0 && p.y > 0) {
                        double w = 0.0;
                        for (const auto &upleftW: upleftWs) {
                            w += upleftW.at<double>(p);
                        }
                        mrf->setNeighbors(vtxIdx, vtxIdx - width - 1, w);
                    }
                    if (p.y > 0) {
                        double w = 0.0;
                        for (const auto &upW: upWs)
                            w += upW.at<double>(p);
                        mrf->setNeighbors(vtxIdx, vtxIdx - width, w);
                    }
                    if (p.x < width - 1 && p.y > 0) {
                        double w = 0.0;
                        for (const auto &uprightW: uprightWs)
                            w = uprightW.at<double>(p);
                        mrf->setNeighbors(vtxIdx, vtxIdx - width + 1, w);
                    }
                }
            }

            mrf->clearAnswer();
            for(p.y=0; p.y<mask.rows; ++p.y){
                for(p.x=0; p.x<mask.cols; ++p.x){
                    mrf->setLabel(p.y*width+p.x, mask.at<int>(p));
                }
            }
            //run alpha-expansion
            mrf->expansion();

            for(p.y=0; p.y < mask.rows; ++p.y){
                for(p.x=0; p.x < mask.cols; ++p.x){
                    mask.at<int>(p) = mrf->getLabel(p.y*width + p.x);
                }
            }
        }

        void mfGrabCut(const std::vector<cv::Mat> &images, cv::Mat &mask, const int iterCount) {
            CHECK(!images.empty());
            CHECK_EQ(images[0].type(), CV_8UC3);
            CHECK_NOTNULL(mask.data);
            Mat hardconstraint;
            const int nLabel = preprocessMask(images[0], mask, hardconstraint);

            std::vector<Mat> compIdxs(images.size());
            for (auto &comid: compIdxs)
                comid.create(images[0].size(), CV_32SC1);

            vector<GMM> gmms((size_t) nLabel);
            initGMMs(images, mask, gmms);

            if (iterCount <= 0)
                return;
            const double gamma = 50;
            const double lambda = 9 * gamma;
            const double beta = calcBeta(images);

            vector<Mat> leftWs(images.size()), upleftWs(images.size()), upWs(images.size()), uprightWs(images.size());
            for (auto v = 0; v < images.size(); ++v)
                calcNWeights(images[v], leftWs[v], upleftWs[v], upWs[v], uprightWs[v], beta, gamma);

            char buffer[128] = {};
            for (int i = 0; i < iterCount; i++) {
                printf("Iter %d\n", i);
                printf("Updating GMM\n");
                for (auto v = 0; v < images.size(); ++v)
                    assignGMMsComponents(images[v], mask, gmms, compIdxs[v]);
                learnGMMs(images, mask, compIdxs, gmms);
                printf("Graph cut...\n");
                runGraphCut(images, mask, hardconstraint, gmms, lambda, leftWs, upleftWs, upWs, uprightWs);

                Mat stepRes = visualizeSegmentation(mask);
                cv::addWeighted(stepRes, 0.8, images[0], 0.2, 0.0, stepRes);
                sprintf(buffer, "iter%03d.png", i);
                imwrite(buffer, stepRes);
            }
        }
    }//namespace video_segment
}//namespace dynamic_stereo