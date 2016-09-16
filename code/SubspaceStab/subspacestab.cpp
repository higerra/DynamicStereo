//
// Created by yanhang on 9/15/16.
//

#include "subspacestab.h"
#include "../base/thread_guard.h"
#include "factorization.h"
#include "warping.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace substab{
    void subSpaceStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output,
                               const SubSpaceStabOption& option){
        FeatureTracks trackMatrix;
        LOG(INFO) << "Computing track matrix";
        Tracking::genTrackMatrix(input, trackMatrix, option.tWindow, option.stride);
        LOG(INFO) << "Total number of tracks: " << trackMatrix.offset.size();
        Eigen::MatrixXd coe, bas, smoothedBas;

        vector<vector<int> > wMatrix(trackMatrix.offset.size());
        for(auto tid=0; tid<wMatrix.size(); ++tid)
            wMatrix[tid].resize(input.size(), 0);

        LOG(INFO) << "Factorization";
        Factorization::movingFactorization(input, trackMatrix, coe, bas, wMatrix, option.tWindow, option.stride);

        MatrixXd reconOri = coe * bas;
        CHECK_EQ(reconOri.rows(), trackMatrix.offset.size()*2);
        CHECK_EQ(reconOri.cols(), input.size());
        //reconstruction error
        double overallError = 0.0;
        double overallCount = 0.0;
        for(auto tid=0; tid<trackMatrix.offset.size(); ++tid){
            const int offset = (int)trackMatrix.offset[tid];
            for(auto v=offset; v<offset+trackMatrix.tracks[tid].size(); ++v){
                if(wMatrix[tid][v] != 1)
                    continue;
                if(v >= input.size() - option.tWindow)
                    continue;
                Vector2d oriPt(trackMatrix.tracks[tid][v-offset].x,trackMatrix.tracks[tid][v-offset].y);
                Vector2d reconPt = reconOri.block(2*tid, v, 2, 1);
                overallCount += 1.0;
                overallError += (reconPt-oriPt).norm();
            }
        }
        LOG(INFO) << "Overall reconstruction error: " << overallError / overallCount;

        LOG(INFO) << "Smoothing";
        Factorization::trackSmoothing(bas, smoothedBas, option.smoothR, -1);
        MatrixXd reconSmo = coe * smoothedBas;
        CHECK_EQ(reconSmo.rows(), trackMatrix.offset.size()*2);
        CHECK_EQ(reconSmo.cols(), input.size());
	
        LOG(INFO) << "Warping";
        GridWarpping warping(input[0].cols, input[0].rows);

        vector<Mat> input2(input.size());
        for(auto i=0; i<input.size(); ++i){
            input2[i] = input[i].clone();
            for(auto y=0; y<input[i].rows; ++y){
                for(auto x=0; x<input[i].cols; ++x){
                    if(input[i].at<Vec3b>(y,x) == Vec3b(0,0,0))
                        input2[i].at<Vec3b>(y,x) = Vec3b(1,1,1);
                }
            }
        }

        output.resize(input.size());
//        const int& num_thread = option.num_thread;
	const int num_thread = 6;

        vector<thread_guard> threads((size_t) num_thread);
        auto threadFunWarp = [&](int threadId) {
            for (auto v = threadId; v < input.size(); v += num_thread) {
		LOG(INFO) << "Frame " << v << " at thread " << threadId;
                vector<Vector2d> pts1, pts2;
                for (auto tid = 0; tid < trackMatrix.offset.size(); ++tid) {
                    const int offset = (int) trackMatrix.offset[tid];
                    if(wMatrix[tid][v] != 1)
                        continue;
                    if (trackMatrix.offset[tid] <= v && offset + trackMatrix.tracks.size() >= v && wMatrix[tid][v]) {
                        pts1.push_back(Vector2d(reconOri(2 * tid, v), reconOri(2 * tid + 1, v)));
                        pts2.push_back(Vector2d(reconSmo(2 * tid, v), reconSmo(2 * tid + 1, v)));
                    }
                }
//		CHECK(!pts1.empty()) << "Frame " << v;
                warping.warpImageCloseForm(input2[v], output[v], pts1, pts2,v);
                if(option.output_drawpoints) {
                    for (auto ftid = 0; ftid < pts2.size(); ++ftid)
                        cv::circle(output[v], cv::Point2d(pts2[ftid][0], pts2[ftid][1]), 1, Scalar(0, 0, 255), 2);
                }
            }
        };

	if(num_thread == 1){
	    threadFunWarp(0);
	}else{
	    for(auto tid=0; tid<threads.size(); ++tid){
		std::thread t(threadFunWarp, tid);
		threads[tid].bind(t);
	    }

	    for(auto& t: threads)
		t.join();
	}

        if(option.output_crop) {
            LOG(INFO) << "Cropping";
            vector<Mat> croped;
            cropImage(output, croped);
            output.swap(croped);
        }
    }

    void cropImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output){
        CHECK(!input.empty());
        const int imgWidth = input[0].cols;
        const int imgHeight = input[0].rows;
        Mat cropMask(imgHeight, imgWidth, CV_32F, Scalar::all(0));
        for(auto y=0; y<imgHeight; ++y){
            for(auto x=0; x<imgWidth; ++x){
                bool has_black = false;
                for(auto v=0; v<input.size(); ++v){
                    if(input[v].at<Vec3b>(y,x) == Vec3b(0,0,0)){
                        has_black = true;
                        break;
                    }
                }
                if(has_black)
                    cropMask.at<float>(y,x) = -1000;
                else
                    cropMask.at<float>(y,x) = 1;
            }
        }
        Mat integralImage;
        cv::integral(cropMask, integralImage, CV_32F);
        Vector4i roi;
        float optValue = -1000 * imgWidth * imgHeight;
        const int stride = 3;
        for(auto x1=0; x1<imgWidth; x1+=stride) {
            for (auto y1 = 0; y1 < imgHeight; y1+=stride) {
                for (auto x2 = x1 + stride; x2 < imgWidth; x2+=stride) {
                    for (auto y2 = y1 + stride; y2 < imgHeight; y2+=stride) {
                        float curValue = integralImage.at<float>(y2, x2) + integralImage.at<float>(y1, x1)
                                         - integralImage.at<float>(y2, x1) - integralImage.at<float>(y1, x2);
                        if(curValue > optValue){
                            optValue = curValue;
                            roi = Vector4i(x1,y1,x2,y2);
                        }
                    }
                }
            }
        }

        output.resize(input.size());
        for(auto i=0; i<output.size(); ++i){
            output[i] = input[i].colRange(roi[0],roi[2]).rowRange(roi[1], roi[3]).clone();
            cv::resize(output[i], output[i], cv::Size(imgWidth, imgHeight));
        }
    }
}//namespace substab
