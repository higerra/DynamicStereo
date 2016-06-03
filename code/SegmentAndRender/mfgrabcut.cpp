//
// Created by yanhang on 6/2/16.
//

#include "dynamicsegment.h"
#include "../external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    using GMMPtr = cv::Ptr<cv::ml::EM>;

    static double  computeBeta(const vector<Mat>& input){
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;
        double count = 0.0;
        double beta = 0;
        for(const auto& img: input) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Vec3d color = img.at<Vec3b>(y, x);
                    if (x > 0) {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                        beta += diff.dot(diff);
                        count += 1.0;
                    }
                    if (y > 0 && x > 0) {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                        beta += diff.dot(diff);
                        count += 1.0;
                    }
                    if (y > 0) {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                        beta += diff.dot(diff);
                        count += 1.0;
                    }
                    if (y > 0 && x < width - 1) {
                        Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                        beta += diff.dot(diff);
                        count += 1.0;
                    }
                }
            }
        }
        if(beta <= std::numeric_limits<double>::epsilon())
            beta = 0;
        else
            beta = 1.0 / (2 * beta/count);
        return beta;
    }

    static void computeAnisotropicDiffusion(const vector<Mat>& input,
                                            Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW,
                                            double beta, double gamma){

    }

    static void estimateGMM(const std::vector<cv::Mat>& images, const cv::Mat& mask, GMMPtr fgGMM, GMMPtr bgGMM){
        vector<Vec3f> posSample, negSample;
        for(auto y=0; y<images[0].rows; ++y){
            for(auto x=0; x<images[0].cols; ++x){
                if(mask.at<uchar>(y,x) == GC_FGD || mask.at<uchar>(y,x) == GC_PR_FGD){
                    for(auto v=0; v<images.size(); ++v){
                        posSample.push_back(static_cast<Vec3f>(images[v].at<Vec3b>(y,x)));
                    }
                }else{
                    for(auto v=0; v<images.size(); ++v){
                        negSample.push_back(static_cast<Vec3f>(images[v].at<Vec3b>(y,x)));
                    }
                }
            }
        }
        Mat posMat((int)posSample.size(), 3, CV_32FC1, &posSample[0][0]);
        Mat negMat((int)negSample.size(), 3, CV_32FC1, &negSample[0][0]);
        CHECK_NOTNULL(fgGMM.get())->trainEM(posMat);
        CHECK_NOTNULL(bgGMM.get())->trainEM(negMat);
    }

    void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask, const int iter){
        //test for grabuct segmentation
        //initial mask
        CHECK(!images.empty());
        const int width = images[0].cols;
        const int height = images[0].rows;
        Mat refImage = images[images.size()/2];

        const double gamma = 50;
        double beta = computeBeta(images);
        const int kCluster = 5;

        Mat compIdx(height, width, CV_32S, Scalar::all(0));

        GMMPtr fgGMM = ml::EM::create(), bgGMM = ml::EM::create();

        estimateGMM(images, mask, fgGMM, bgGMM);

        Mat result(mask.size(), CV_8UC1, Scalar::all(GC_PR_BGD));
        for(auto ii=0; ii<iter; ++ii){
            //assign each pixel a component

            //re-estimate GMM

            //update mask by graph cup
        }
        //estimate GMM
//				cv::Ptr<cv::ml::EM> gmm_positive = cv::ml::EM::create();
//
//				vector<Vector3d> psamples;
//				//collect classifier sample
//				for (auto y = top; y < top + roih; ++y) {
//					for (auto x = left; x < left + roiw; ++x) {
//						if (pLabel[(y + top) * width + x + left] == l) {
//							for (auto v = 0; v < input.size(); ++v) {
//								Vec3b pix = input[v].at<Vec3b>(y + top, x + left);
//								psamples.push_back(Vector3d((double) pix[0], (double) pix[1], (double) pix[2]));
//							}
//						}
//					}
//				}
//				Mat ptrainSample((int)psamples.size(), 3, CV_64F);
//				for(auto i=0; i<psamples.size(); ++i){
//					ptrainSample.at<double>(i,0) = psamples[i][0];
//					ptrainSample.at<double>(i,1) = psamples[i][1];
//					ptrainSample.at<double>(i,2) = psamples[i][2];
//				}
//
//				cout << "Estimating foreground color model..." << endl;
//				gmm_positive->trainEM(ptrainSample);
//
//
//				vector<double> unary;
//				assignColorTerm(input, gmm_positive, gmm_negative, unary);

    }
}//namespace dynamic_stereo

