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
#include "gcgraph.hpp"

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

            GMM(Mat &_model);

            double operator()(const Vec3d &color) const;

            double operator()(int ci, const Vec3d &color) const;

            int whichComponent(const Vec3d color) const;

            void initLearning();

            void addSample(int ci, const Vec3d color);

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

        GMM::GMM(Mat &_model) {
            const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
            if (_model.empty()) {
                _model.create(1, modelSize * componentsCount, CV_64FC1);
                _model.setTo(Scalar(0));
            } else if ((_model.type() != CV_64FC1) || (_model.rows != 1) ||
                       (_model.cols != modelSize * componentsCount))
                CV_Error(CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");

            model = _model;

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

        void GMM::addSample(int ci, const Vec3d color) {
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
        static void checkMask(const Mat &img, const Mat &mask) {
            if (mask.empty())
                CV_Error(CV_StsBadArg, "mask is empty");
            if (mask.type() != CV_8UC1)
                CV_Error(CV_StsBadArg, "mask must have CV_8UC1 type");
            if (mask.cols != img.cols || mask.rows != img.rows)
                CV_Error(CV_StsBadArg, "mask must have as many rows and cols as img");
            for (int y = 0; y < mask.rows; y++) {
                for (int x = 0; x < mask.cols; x++) {
                    uchar val = mask.at<uchar>(y, x);
                    if (val != GC_BGD && val != GC_FGD && val != GC_PR_BGD && val != GC_PR_FGD)
                        CV_Error(CV_StsBadArg, "mask element value must be equel"
                                "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD");
                }
            }
        }


        /*
      Initialize GMM background and foreground models using kmeans algorithm.
    */
        static void initGMMs(const vector<Mat> &images, const Mat &mask, GMM &bgdGMM, GMM &fgdGMM) {
            const int kMeansItCount = 10;
            const int kMeansType = KMEANS_PP_CENTERS;

            Mat bgdLabels, fgdLabels;
            std::vector<Vec3f> bgdSamples, fgdSamples;
            Point p;
            for (const auto &img: images) {
                for (p.y = 0; p.y < img.rows; p.y++) {
                    for (p.x = 0; p.x < img.cols; p.x++) {
                        if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                            bgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
                        else // GC_FGD | GC_PR_FGD
                            fgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
                    }
                }
            }
            CHECK(!bgdSamples.empty());
            CHECK(!fgdSamples.empty());
            Mat _bgdSamples((int) bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
            kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
                   TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
            Mat _fgdSamples((int) fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
            kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
                   TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

            bgdGMM.initLearning();
            for (int i = 0; i < (int) bgdSamples.size(); i++)
                bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
            bgdGMM.endLearning();

            fgdGMM.initLearning();
            for (int i = 0; i < (int) fgdSamples.size(); i++)
                fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
            fgdGMM.endLearning();
        }

/*
  Assign GMMs components for each pixel.
*/
        static void assignGMMsComponents(const Mat &img, const Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM,
                                         Mat &compIdxs) {
            Point p;
            for (p.y = 0; p.y < img.rows; p.y++) {
                for (p.x = 0; p.x < img.cols; p.x++) {
                    Vec3d color = img.at<Vec3b>(p);
                    compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                                          bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
                }
            }
        }

/*
  Learn GMMs parameters.
*/
        static void learnGMMs(const vector<Mat> &images, const Mat &mask, const vector<Mat> &compIdxs_all, GMM &bgdGMM,
                              GMM &fgdGMM) {
            bgdGMM.initLearning();
            fgdGMM.initLearning();
            Point p;
            for (auto v = 0; v < images.size(); ++v) {
                const Mat &img = images[v];
                const Mat &compIdxs = compIdxs_all[v];
                for (int ci = 0; ci < GMM::componentsCount; ci++) {
                    for (p.y = 0; p.y < img.rows; p.y++) {
                        for (p.x = 0; p.x < img.cols; p.x++) {
                            if (compIdxs.at<int>(p) == ci) {
                                if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                                    bgdGMM.addSample(ci, img.at<Vec3b>(p));
                                else
                                    fgdGMM.addSample(ci, img.at<Vec3b>(p));
                            }
                        }
                    }
                }
            }
            bgdGMM.endLearning();
            fgdGMM.endLearning();
        }

/*
  Construct GCGraph
*/
        static void constructGCGraph(const vector<Mat> &images, const Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM,
                                     double lambda,
                                     const vector<Mat> &leftWs, const vector<Mat> &upleftWs, const vector<Mat> &upWs,
                                     const vector<Mat> &uprightWs,
                                     GCGraph<double> &graph) {
            CHECK(!images.empty());
            const int width = images[0].cols;
            const int height = images[0].rows;
            int vtxCount = width * height;
            int edgeCount = 2 * (4 * width * height - 3 * (width + height) + 2);
            graph.create(vtxCount, edgeCount);
            Point p;
            for (p.y = 0; p.y < height; p.y++) {
                for (p.x = 0; p.x < width; p.x++) {
                    // add node
                    int vtxIdx = graph.addVtx();
                    double fromSource = 0, toSink = 0;
                    // set t-weights
                    if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                        for (const auto &img: images) {
                            Vec3b color = img.at<Vec3b>(p);
                            fromSource -= log(bgdGMM(color));
                            toSink -= log(fgdGMM(color));
                        }
                    } else if (mask.at<uchar>(p) == GC_BGD) {
                        fromSource = 0;
                        toSink += lambda * (double) images.size();
                    } else // GC_FGD
                    {
                        fromSource += lambda * (double) images.size();
                        toSink = 0;
                    }

                    graph.addTermWeights(vtxIdx, fromSource, toSink);

                    // set n-weights
                    if (p.x > 0) {
                        double w = 0.0;
                        for (const auto &leftW: leftWs) {
                            w += leftW.at<double>(p);
                        }
                        graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
                    }
                    if (p.x > 0 && p.y > 0) {
                        double w = 0.0;
                        for (const auto &upleftW: upleftWs) {
                            w += upleftW.at<double>(p);
                        }
                        graph.addEdges(vtxIdx, vtxIdx - width - 1, w, w);
                    }
                    if (p.y > 0) {
                        double w = 0.0;
                        for (const auto &upW: upWs)
                            w += upW.at<double>(p);
                        graph.addEdges(vtxIdx, vtxIdx - width, w, w);
                    }
                    if (p.x < width - 1 && p.y > 0) {
                        double w = 0.0;
                        for (const auto &uprightW: uprightWs)
                            w = uprightW.at<double>(p);
                        graph.addEdges(vtxIdx, vtxIdx - width + 1, w, w);
                    }
                }
            }
        }

/*
  Estimate segmentation using MaxFlow algorithm
*/
        static void estimateSegmentation(GCGraph<double> &graph, Mat &mask) {
            graph.maxFlow();
            Point p;
            for (p.y = 0; p.y < mask.rows; p.y++) {
                for (p.x = 0; p.x < mask.cols; p.x++) {
                    if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                        if (graph.inSourceSegment(p.y * mask.cols + p.x /*vertex index*/ ))
                            mask.at<uchar>(p) = GC_PR_FGD;
                        else
                            mask.at<uchar>(p) = GC_PR_BGD;
                    }
                }
            }
        }

        void mfGrabCut(const std::vector<cv::Mat> &images, cv::Mat &mask, const int iterCount) {
            CHECK(!images.empty());
            CHECK_NOTNULL(mask.data);

            Mat bgdModel, fgdModel;
            GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
            std::vector<Mat> compIdxs(images.size());
            for (auto &comid: compIdxs)
                comid.create(images[0].size(), CV_32SC1);

            checkMask(images[0], mask);
            initGMMs(images, mask, bgdGMM, fgdGMM);

            if (iterCount <= 0)
                return;
            const double gamma = 50;
            const double lambda = 9 * gamma;
            const double beta = calcBeta(images);

            vector<Mat> leftWs(images.size()), upleftWs(images.size()), upWs(images.size()), uprightWs(images.size());
            for (auto v = 0; v < images.size(); ++v)
                calcNWeights(images[v], leftWs[v], upleftWs[v], upWs[v], uprightWs[v], beta, gamma);

            for (int i = 0; i < iterCount; i++) {
                GCGraph<double> graph;
                for (auto v = 0; v < images.size(); ++v)
                    assignGMMsComponents(images[v], mask, bgdGMM, fgdGMM, compIdxs[v]);
                learnGMMs(images, mask, compIdxs, bgdGMM, fgdGMM);
                constructGCGraph(images, mask, bgdGMM, fgdGMM, lambda, leftWs, upleftWs, upWs, uprightWs, graph);
                estimateSegmentation(graph, mask);
            }
        }
    }//namespace video_segment
}//namespace dynamic_stereo