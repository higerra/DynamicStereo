//
// Created by yanhang on 9/13/16.
//

#include "stabilization.h"
#include "../base/opticalflow.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{

    void flowStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const double lambda,
                           const cv::InputArray inputMask){
        CHECK(!input.empty());
        output.resize(input.size());

        //Initalize mask
        Mat mask;
        if(!inputMask.empty()) {
            mask = inputMask.getMat();
            CHECK_EQ(mask.size(), input[0].size());
            CHECK_EQ(mask.type(), CV_8UC1);
        } else{
            mask.create(input[0].size(), CV_8UC1);
            mask.setTo(cv::Scalar(255));
        }

        int kPix = 0;
        for(auto i=0; i<mask.cols * mask.rows; ++i){
            if(mask.data[i] > (uchar)200)
                kPix++;
        }

        //compute optical flow
        FlowEstimatorGPU flowEstimator;
        vector<FlowFrame> flows(input.size());
        //Note that the backward flows are computed. flows[v] stores flow from v -> v-1
        for(auto v=1; v<input.size(); ++v){
            Mat img1, img2;
            cvtColor(input[v], img1, CV_BGR2GRAY);
            cvtColor(input[v-1], img2, CV_BGR2GRAY);
            flowEstimator.estimate(img1, img2, flows[v], 1);

//            Mat flowVis;
//            flow_util::visualizeFlow(flows[v], flowVis);
//            imshow("flow", flowVis);
//            waitKey(0);
        }

        Eigen::MatrixXd flowMat(kPix * 2, (int)input.size() - 1);
        int rowInd = 0;
        for(auto y=0; y<mask.rows; ++y){
            for(auto x=0; x<mask.cols; ++x){
                if(mask.at<uchar>(y,x) < (uchar)200)
                    continue;
                for(auto v=1; v<input.size(); ++v){
                    Vector2d fv = flows[v].getFlowAt(x,y);
                    flowMat(2 * rowInd, v-1) = fv[0];
                    flowMat(2 * rowInd + 1, v-1) = fv[1];
                }
                rowInd++;
            }
        }

        CHECK_EQ(rowInd, kPix);

        //run RPCA
        Eigen::MatrixXd resFlow;

        //pass through
        resFlow = flowMat;

        //warp by result flow
        output[0] = input[0].clone();
        for(auto v=1; v<input.size(); ++v){
            rowInd = 0;
            FlowFrame curFlow(input[v].cols, input[v].rows);
            for(auto y=0; y<mask.rows; ++y){
                for(auto x=0; x<mask.cols; ++x){
                    if(mask.at<uchar>(y,x) < (uchar)200)
                        continue;
                    Vector2d fv(resFlow(2 * rowInd, v-1), resFlow(2 * rowInd + 1, v-1));
                    //Note: the warping flow should be the difference between result flow and original flow
                    curFlow.setValue(x,y, fv - flows[v].getFlowAt(x,y));
                    rowInd++;
                }
            }
        }
    }
}//namespace dynamic_stereo
