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
        for(auto v=1; v<input.size(); ++v){
            flowEstimator.estimate(input[v-1], input[v], flows[v], 1);
        }
        Eigen::MatrixXd xPos(kPix, (int)input.size() - 1);
        Eigen::MatrixXd yPos(kPix, (int)input.size() - 1);
        for(auto y=0; y<mask.rows; ++y){
            for(auto x=0; x<mask.cols; ++x){

            }
        }

    }
}//namespace dynamic_stereo
