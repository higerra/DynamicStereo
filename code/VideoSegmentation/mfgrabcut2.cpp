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
#include <Eigen/Eigen>
#include <limits>
#include <memory>

#include <opengm/opengm.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>

#include "../external/MRF2.2/GCoptimization.h"
#include "videosegmentation.h"
#include "colorGMM.h"

using namespace cv;
using namespace std;
using namespace Eigen;
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

        //Ungly global variable... This is really used on every functions, so
        //declare here to avoid massive function parameter...
        static bool hasInvalid = false;

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
                        if(cv::norm(color) < numeric_limits<double>::min())
                            continue;
                        if (x > 0) // left
                        {
                            Vec3d neiColor = (Vec3d) img.at<Vec3b>(y,x-1);
                            if((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                                Vec3d diff = color - neiColor;
                                beta += diff.dot(diff);
                                count += 1.0;
                            }
                        }
                        if (y > 0 && x > 0) // upleft
                        {
                            Vec3d neiColor = (Vec3d) img.at<Vec3b>(y-1, x-1);
                            if((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                                Vec3d diff = color - neiColor;
                                beta += diff.dot(diff);
                                count += 1.0;
                            }
                        }
                        if (y > 0) // up
                        {
                            Vec3d neiColor = (Vec3d) img.at<Vec3b>(y-1, x);
                            if((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                                Vec3d diff = color - neiColor;
                                beta += diff.dot(diff);
                                count += 1.0;
                            }
                        }
                        if (y > 0 && x < img.cols - 1) // upright
                        {
                            Vec3d neiColor = (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                            if ((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                                Vec3d diff = color - neiColor;
                                beta += diff.dot(diff);
                                count += 1.0;
                            }
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

            leftW.setTo(Scalar::all(0));
            upW.setTo(Scalar::all(0));
            upleftW.setTo(Scalar::all(0));
            uprightW.setTo(Scalar::all(0));

            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    Vec3d color = img.at<Vec3b>(y, x);
                    if(hasInvalid && cv::norm(color) < numeric_limits<double>::min()){
                        continue;
                    }
                    if (x - 1 >= 0) {
                        Vec3d neiColor = (Vec3d) img.at<Vec3b>(y, x - 1);
                        if ((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                            Vec3d diff = color - neiColor;
                            leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                        }
                    }
                    if (x - 1 >= 0 && y - 1 >= 0){
                        Vec3d neiColor = (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                        if ((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                            Vec3d diff = color - neiColor;
                            upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                        }
                    }
                    if (y - 1 >= 0){
                        Vec3d neiColor = (Vec3d) img.at<Vec3b>(y - 1, x);
                        if ((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                            Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                            upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                        }
                    }
                    if (x + 1 < img.cols && y - 1 >= 0){
                        Vec3d neiColor = (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                        if ((!hasInvalid) || (hasInvalid && cv::norm(neiColor) > numeric_limits<double>::min())) {
                            Vec3d diff = color - neiColor;
                            uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                        }
                    }
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
        static void initGMMs(const vector<Mat> &images, const Mat &mask, vector<ColorGMM>& gmms) {
            CHECK(!gmms.empty());
            const int kMeansItCount = 10;
            const int kMeansType = KMEANS_PP_CENTERS;
            const int nLabel = gmms.size();

            std::vector<std::vector<Vec3f> > samples(gmms.size());
            Point p;
            for (const auto &img: images) {
                for (p.y = 0; p.y < img.rows; p.y++) {
                    for (p.x = 0; p.x < img.cols; p.x++) {
                        Vec3b c = img.at<Vec3b>(p);
                        if (!hasInvalid || (hasInvalid && (cv::norm(c) > 0))) {
                            const int comId = mask.at<int>(p);
                            samples[comId].push_back((Vec3f) img.at<Vec3b>(p));
                        }
                    }
                }
            }
            for(const auto& s: samples)
                CHECK(!s.empty());
#pragma omp parallel for
            for(auto l=0; l<samples.size(); ++l){
                Mat bestLabels;
                Mat sampleMat((int) samples[l].size(), 3, CV_32FC1, &samples[l][0][0]);
                kmeans(sampleMat, ColorGMM::componentsCount, bestLabels, TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

                gmms[l].initLearning();
                for(auto i=0; i<samples[l].size(); ++i)
                    gmms[l].addSample(bestLabels.at<int>(i,0), samples[l][i]);
                gmms[l].endLearning();
            }
        }
/*
  Assign GMMs components for each pixel.
*/
        //if a pixel is invalid (black), the component id will be set to -1
        static void assignGMMsComponents(const Mat &img, const Mat &mask, const std::vector<ColorGMM>& gmms,
                                         Mat &compIdxs) {
            for (auto y = 0; y < img.rows; ++y) {
                for (auto x = 0; x < img.cols; ++x) {
                    Vec3f color = (Vec3f)img.at<Vec3b>(y,x);
                    if(hasInvalid && cv::norm(color) < numeric_limits<float>::min()){
                        compIdxs.at<int>(y,x) = -1;
                    }else {
                        const int lid = mask.at<int>(y, x);
                        compIdxs.at<int>(y, x) = gmms[lid].whichComponent(color);
                    }
                }
            }
        }

/*
  Learn GMMs parameters.
*/
        static void learnGMMs(const vector<Mat> &images, const Mat &mask, const vector<Mat> &compIdxs_all, vector<ColorGMM>& gmms) {
            for(auto& g: gmms)
                g.initLearning();
            for(auto v=0; v<images.size(); ++v){
                const Mat &img = images[v];
                const Mat &compIdxs = compIdxs_all[v];
                for (auto y = 0; y < img.rows; ++y) {
                    for (auto x = 0; x < img.cols; ++x) {
                        const int comId = compIdxs.at<int>(y,x);
                        if(comId < 0)
                            continue;
                        const int lid = mask.at<int>(y,x);
                        gmms[lid].addSample(comId, (Vec3f) img.at<Vec3b>(y,x));
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
                                const std::vector<ColorGMM>& gmms,
                                double lambda,
                                const vector<Mat> &leftWs, const vector<Mat> &upleftWs, const vector<Mat> &upWs,
                                const vector<Mat> &uprightWs){
            CHECK(!images.empty());
            printf("Using MRF2.2\n");
            const int width = images[0].cols;
            const int height = images[0].rows;
            const int nLabel = (int)gmms.size();
            int vtxCount = width * height;

            //the energy if divided by ratio, to avoid exceed the limit of int
            const double ratio = 1000.0;

            vector<double> MRF_data((size_t) nLabel * vtxCount, 0.0);
            vector<double> MRF_smoothess((size_t) nLabel * nLabel, 1.0 / ratio);
            for(auto l1=0; l1 < nLabel; ++l1){
                for(auto l2=0; l2 < nLabel; ++l2){
                    if(l1 == l2)
                        MRF_smoothess[l1*nLabel+l2] = 0.0;
                }
            }

            float start_t = (float)cv::getTickCount();
            printf("data term...\n");
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
                                if(!hasInvalid || (hasInvalid && (cv::norm(color) > numeric_limits<double>::min())))
                                    e -= log(gmms[l](color));
                            }
                        }
                        MRF_data[vtxIdx * nLabel + l] = e / ratio;
                    }
                }
            }

            DataCost* dataCost = new DataCost(MRF_data.data());
            SmoothnessCost* smoothnessCost = new SmoothnessCost(MRF_smoothess.data());
            EnergyFunction* energy_function = new EnergyFunction(dataCost, smoothnessCost);
            Expansion* mrf = new Expansion(vtxCount, nLabel, energy_function);

            mrf->initialize();

            printf("Smoothness term...\n");
            for(auto y = 0; y < height; ++y){
                for(auto x=0; x < width; ++x){
                    //assign smoothness weight
                    const int vtxIdx = y * width + x;
                    if (x > 0) {
                        double w = 0.0;
                        for (const auto &leftW: leftWs) {
                            w += leftW.at<double>(y,x );
                        }
                        mrf->setNeighbors(vtxIdx, vtxIdx-1, w);
                    }
                    if (x > 0 && y > 0) {
                        double w = 0.0;
                        for (const auto &upleftW: upleftWs) {
                            w += upleftW.at<double>(y,x);
                        }
                        mrf->setNeighbors(vtxIdx, vtxIdx - width - 1, w);
                    }
                    if (y > 0) {
                        double w = 0.0;
                        for (const auto &upW: upWs)
                            w += upW.at<double>(y,x);
                        mrf->setNeighbors(vtxIdx, vtxIdx - width, w);
                    }
                    if (x < width - 1 && y > 0) {
                        double w = 0.0;
                        for (const auto &uprightW: uprightWs)
                            w = uprightW.at<double>(y,x);
                        mrf->setNeighbors(vtxIdx, vtxIdx - width + 1, w);
                    }
                }
            }

            for (auto y = 0; y < mask.rows; ++y) {
                for (auto x = 0; x < mask.cols; ++x) {
                    mrf->setLabel(y * width + x, mask.at<int>(y, x));
                }
            }
            mrf->clearAnswer();
            //run alpha-expansion
            printf("Inital energy: %.3f\n", mrf->totalEnergy());
            printf("Solving...\n");
            start_t = (float)getTickCount();
            mrf->expansion();
            printf("Done, final energy: %.3f, time usage: %.2fs\n", mrf->totalEnergy(),
                   ((float)getTickCount() - start_t) / (float)getTickFrequency());


            for(auto y=0; y < mask.rows; ++y){
                for(auto x=0; x < mask.cols; ++x){
                    mask.at<int>(y,x) = mrf->getLabel(y*width + x);
                }
            }
            delete mrf;
            delete energy_function;
            delete smoothnessCost;
            delete dataCost;

        }

        static void runGraphCutOpenGM(const vector<Mat> &images, Mat &mask, const Mat& hardConstraint,
                                const std::vector<ColorGMM>& gmms,
                                double lambda,
                                const vector<Mat> &leftWs, const vector<Mat> &upleftWs, const vector<Mat> &upWs,
                                const vector<Mat> &uprightWs){
            CHECK(!images.empty());
            printf("Using OpenGM\n");
            //typedef
            using GMSpace = opengm::SimpleDiscreteSpace<size_t, size_t>;
            using UnaryFunction = opengm::ExplicitFunction<double>;
            using PairwiseFunction = opengm::PottsFunction<double>;
            using GMModel = opengm::GraphicalModel<double, opengm::Adder,
                    OPENGM_TYPELIST_2(UnaryFunction, PairwiseFunction), GMSpace>;

            const int width = images[0].cols;
            const int height = images[0].rows;
            const int nLabel = (int)gmms.size();
            int vtxCount = width * height;
            const double ratio = 1000;

            GMSpace space(vtxCount, (size_t)nLabel);
            GMModel model(space);

            //the energy if divided by ratio, to avoid exceed the limit of int
            float start_t = (float)cv::getTickCount();
            printf("data term...\n");

            vector<double> MRF_data(vtxCount * nLabel, 0.0);
#pragma omp parallel for
            for(auto y=0; y < height; ++y){
                for(auto x = 0; x < width; ++x){
                    const int vtxIdx = y * width + x;
                    //assign data term
                    for(size_t l=0; l<nLabel; ++l){
                        double e = 0.0;
                        if(hardConstraint.at<uchar>(y,x) > (uchar)200){
                            if(l != mask.at<int>(y,x))
                                e = lambda * (double)images.size();
                        }else {
                            for (const auto &img: images) {
                                Vec3d color = (Vec3d) img.at<Vec3b>(y,x);
                                if(!hasInvalid || (hasInvalid && (cv::norm(color) > numeric_limits<double>::min())))
                                    e -= std::max(log(gmms[l](color)), 0.0);
                            }
                        }
                        MRF_data[vtxIdx * nLabel + l] = e / ratio;
                    }

                }
            }

            for(auto y=0; y<height; ++y){
                for(auto x=0; x<width; ++x){
                    vector<size_t> shape{(size_t)nLabel};
                    const size_t vtxIdx = y * width + x;
                    UnaryFunction unary(shape.begin(), shape.end());
                    for(auto l=0; l<nLabel; ++l){
                        unary(l) = MRF_data[vtxIdx * nLabel + l];
                    }
                    GMModel::FunctionIdentifier fid = model.addFunction(unary);
                    vector<size_t> vid{(size_t)vtxIdx};
                    model.addFactor(fid, vid.begin(), vid.end());
                }
            }
            MRF_data.clear();

            printf("Smoothness term...\n");
            for(auto y = 0; y < height; ++y){
                for(auto x=0; x < width; ++x){
                    //assign smoothness weight
                    const int vtxIdx = y * width + x;
                    if (x > 0) {
                        double w = 0.0;
                        for (const auto &leftW: leftWs) {
                            w += leftW.at<double>(y,x );
                        }
                        vector<size_t> vid{(size_t)vtxIdx - 1, (size_t)vtxIdx};
                        PairwiseFunction pf((size_t)nLabel, (size_t)nLabel, 0.0, w / ratio);
                        GMModel::FunctionIdentifier fid = model.addFunction(pf);
                        model.addFactor(fid, vid.begin(), vid.end());
                    }
                    if (x > 0 && y > 0) {
                        double w = 0.0;
                        for (const auto &upleftW: upleftWs) {
                            w += upleftW.at<double>(y,x);
                        }
                        vector<size_t> vid{(size_t)vtxIdx - width - 1, (size_t)(vtxIdx)};
                        PairwiseFunction pf((size_t)nLabel, (size_t)nLabel, 0.0, w / ratio);
                        GMModel::FunctionIdentifier fid = model.addFunction(pf);
                        model.addFactor(fid, vid.begin(), vid.end());
                    }
                    if (y > 0) {
                        double w = 0.0;
                        for (const auto &upW: upWs)
                            w += upW.at<double>(y,x);
                        vector<size_t> vid{(size_t)vtxIdx - width, (size_t)(vtxIdx)};
                        PairwiseFunction pf((size_t)nLabel, (size_t)nLabel, 0.0, w / ratio);
                        GMModel::FunctionIdentifier fid = model.addFunction(pf);
                        model.addFactor(fid, vid.begin(), vid.end());

                    }
                    if (x < width - 1 && y > 0) {
                        double w = 0.0;
                        for (const auto &uprightW: uprightWs)
                            w = uprightW.at<double>(y,x);
                        vector<size_t> vid{(size_t)vtxIdx - width + 1, (size_t)(vtxIdx)};
                        PairwiseFunction pf((size_t)nLabel, (size_t)nLabel, 0.0, w / ratio);
                        GMModel::FunctionIdentifier fid = model.addFunction(pf);
                        model.addFactor(fid, vid.begin(), vid.end());
                    }
                }
            }

            using MinStCutType = opengm::external::MinSTCutKolmogorov<size_t, double>;
            using MinGraphCut = opengm::GraphCut<GMModel, opengm::Minimizer, MinStCutType>;
            using MinAlphaExpansion = opengm::AlphaExpansion<GMModel, MinGraphCut>;

            MinAlphaExpansion expansion(model);
            //run alpha-expansion
            printf("Solving...\n");
            printf("Inital energy: %.3f\n", expansion.value());
            start_t = (float)getTickCount();
            expansion.infer();
            printf("Done, final energy: %.3f, time usage: %.2fs\n", expansion.value(),
                   ((float)getTickCount() - start_t) / (float)getTickFrequency());

            vector<size_t> result;
            expansion.arg(result);
            CHECK_EQ(result.size(), mask.rows * mask.cols);
            for(auto y=0; y < mask.rows; ++y){
                for(auto x=0; x < mask.cols; ++x){
                    mask.at<int>(y,x) = result[y*width + x];
                }
            }
        }

        void mfGrabCut(const std::vector<cv::Mat> &images, cv::Mat &mask,
                       const bool hasInvalid_, const int iterCount) {
            CHECK(!images.empty());
            CHECK_EQ(images[0].type(), CV_8UC3);
            CHECK_NOTNULL(mask.data);
            hasInvalid = hasInvalid_;

            Mat hardconstraint;
            const int nLabel = preprocessMask(images[0], mask, hardconstraint);
            std::vector<Mat> compIdxs(images.size());
            for (auto &comid: compIdxs)
                comid.create(images[0].size(), CV_32SC1);

            printf("Init...\n");
            vector<ColorGMM> gmms((size_t) nLabel);
            initGMMs(images, mask, gmms);
            if (iterCount <= 0)
                return;
            const double gamma = 50;
            const double lambda = 9 * gamma;
            const double beta = calcBeta(images);

            //debug: test GPU feature
            {
//                printf("Testing on GPU\n");
//                float start_t = (float)cv::getTickCount();
//                vector<Mat> gpuOutput;
//                gmms[0].assignComponentCuda(images, gpuOutput, 200);
//                printf("Cuda time usage: %.5f\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());
//
//                vector<Mat> cpuOutput(images.size());
//                for(auto& o: cpuOutput)
//                    o.create(images[0].size(), CV_32SC1);
//                printf("Testing on CPU\n");
//                start_t = (float)cv::getTickCount();
//#pragma omp parallel for
//                for(auto v=0; v<images.size(); ++v) {
//                    for (auto y = 0; y < images[0].rows; ++y) {
//                        for (auto x = 0; x < images[0].cols; ++x) {
//                            cpuOutput[v].at<int>(y, x) = gmms[0].whichComponent((Vec3f)images[v].at<Vec3b>(y,x));
//                        }
//                    }
//                }
//                printf("CPU time usage: %.5f\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());
//                CHECK_EQ(gpuOutput.size(), cpuOutput.size());
//                float cpuGpuDiff = 0.0;
//                for(auto v=0; v<images.size(); ++v)
//                    cpuGpuDiff += cv::norm(gpuOutput[v] - cpuOutput[v]);
//                printf("Result diff: %.1f\n", cpuGpuDiff);
            }

            vector<Mat> leftWs(images.size()), upleftWs(images.size()), upWs(images.size()), uprightWs(images.size());
            for (auto v = 0; v < images.size(); ++v)
                calcNWeights(images[v], leftWs[v], upleftWs[v], upWs[v], uprightWs[v], beta, gamma);

            char buffer[128] = {};
            for (int i = 0; i < iterCount; i++) {
                printf("Iter %d\n", i);
                printf("Assigning component...\n");
#pragma omp parallel for
                for (auto v = 0; v < images.size(); ++v)
                    assignGMMsComponents(images[v], mask, gmms, compIdxs[v]);
                learnGMMs(images, mask, compIdxs, gmms);
                printf("graph cut...\n");
                //runGraphCut(images, mask, hardconstraint, gmms, lambda, leftWs, upleftWs, upWs, uprightWs);
                runGraphCutOpenGM(images, mask, hardconstraint, gmms, lambda, leftWs, upleftWs, upWs, uprightWs);

                Mat stepRes = visualizeSegmentation(mask);
                cv::addWeighted(stepRes, 0.8, images[0], 0.2, 0.0, stepRes);
            }
        }
    }//namespace video_segment
}//namespace dynamic_stereo
