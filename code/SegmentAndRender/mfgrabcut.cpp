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

    static void assignGMMComponet(const std::vector<cv::Mat>& images, const cv::Mat& mask,
                             GMMPtr fgGMM, GMMPtr bgGMM, std::vector<cv::Mat>& compIdx){
        CHECK_EQ(compIdx.size(), images.size());
        for(auto v=0; v<images.size(); ++v){
            CHECK_EQ(compIdx[v].type(), CV_32S);
            for(auto y=0; y<images[v].rows; ++y){
                for(auto x=0; x<images[v].cols; ++x){
                    Vec3f sample = static_cast<Vec3f>(images[v].at<Vec3b>(y,x));
                    Mat probMat;
                    Vec2d res;
                    if(mask.at<uchar>(y,x) == GC_FGD || mask.at<uchar>(y,x) == GC_PR_FGD)
                        res = fgGMM->predict2(sample, probMat);
                    else
                        res = bgGMM->predict2(sample, probMat);
                    int comId = (int)res[1];
                    CHECK_GE(comId, 0);
                    CHECK_LT(comId, fgGMM->getClustersNumber());
                    compIdx[v].at<int>(y,x) = (int)res[1];
                }
            }
        }
    }

    static Mat MRFSegment(const std::vector<cv::Mat>& images, GMMPtr fgGMM, GMMPtr bgGMM, const std::vector<Mat>& compIdx){
        CHECK_NOTNULL(fgGMM.get());
        CHECK_NOTNULL(bgGMM.get());
        CHECK(!images.empty());
        const int width = images[0].cols;
        const int height = images[0].rows;

        //assign data term
        vector<double> data_cost((size_t)width * height * 2, 0.0);
        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){

            }
        }
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

        vector<Mat> compIdx(images.size());
        for(auto& idxMat: compIdx)
            idxMat.create(height, width, CV_32S);

        GMMPtr fgGMM = ml::EM::create();
        GMMPtr bgGMM = ml::EM::create();

        //initialize GMM
        estimateGMM(images, mask, fgGMM, bgGMM);

        Mat result(mask.size(), CV_8UC1, Scalar::all(GC_PR_BGD));
        for(auto ii=0; ii<iter; ++ii){
            //assign each pixel a component
            assignGMMComponet(images, mask, fgGMM, bgGMM, compIdx);
            //re-estimate GMM
            estimateGMM(images, mask, fgGMM, bgGMM);
            //update mask by graph cut

        }
    }
}//namespace dynamic_stereo

