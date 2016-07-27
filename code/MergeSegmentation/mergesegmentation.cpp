//
// Created by yanhang on 7/27/16.
//

#include "mergesegmentation.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo {
    void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output){
        CHECK(!input.empty());
        output.create(input[0].size(), CV_32FC1);
        output.setTo(cv::Scalar::all(0));
        for (auto i =0; i<input.size(); ++i) {
            Mat edge_sobel(input[i].size(), CV_32FC1, Scalar::all(0));
            Mat gray, gx, gy;
            cvtColor(input[i], gray, CV_BGR2GRAY);
            cv::blur(gray, gray, cv::Size(3,3));
            cv::Sobel(gray, gx, CV_32F, 1, 0);
            cv::Sobel(gray, gy, CV_32F, 0, 1);
            for(auto y=0; y<gray.rows; ++y){
                for(auto x=0; x<gray.cols; ++x){
                    float ix = gx.at<float>(y,x);
                    float iy = gy.at<float>(y,x);
                    edge_sobel.at<float>(y,x) = std::sqrt(ix*ix+iy*iy+FLT_EPSILON);
                }
            }
            output += edge_sobel;
        }

        double maxedge, minedge;
        cv::minMaxLoc(output, &minedge, &maxedge);
        if(maxedge > 0)
            output /= maxedge;
    }
}//namespace dynamic_stereo