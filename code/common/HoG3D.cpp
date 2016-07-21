//
// Created by yanhang on 7/20/16.
//

#include "HoG3D.h"
using namespace std;
namespace cv {
    CVHoG3D::CVHoG3D(const int ss_, const int sr_, const int M_, const int N_, const int kSubBlock_)
            : M(M_), N(N_), kSubBlock(kSubBlock_), sigma_s(ss_),  sigma_r(sr_){}

    void CVHoG3D::compute(const _InputArray &image, std::vector<KeyPoint> &keypoints,
                          const _OutputArray &descriptors) {
        CHECK(!keypoints.empty());
        vector<Mat> input;
        image.getMatVector(input);
        CHECK_GE(input.size(), sigma_r);
        dynamic_stereo::Feature::HoG3D hog3D(M,N,kSubBlock);

        descriptors.create((int)keypoints.size(), hog3D.getDim(), CV_32FC1);
        Mat& descriptors_ = descriptors.getMatRef();

        for(auto i=0; i<keypoints.size(); ++i){
            vector<Mat> subVideo((size_t)sigma_r);
            const Point2f& pt = keypoints[i].pt;
            cv::Rect roi((int)pt.x - sigma_s/2 + 1, (int)pt.y-sigma_s/2 + 1, sigma_s-1, sigma_s-1);
            const int startId = keypoints[i].octave - sigma_r / 2;
            const int endId = startId + sigma_r - 1;
            //boundary check
            CHECK_GE(roi.x, 0);
            CHECK_GE(roi.y, 0);
            CHECK_LT(roi.br().x, input[0].cols);
            CHECK_LT(roi.br().y, input[0].rows);
            CHECK_GE(startId, 0);
            CHECK_LT(endId, input.size());

            for(auto v=startId; v<=endId; ++v)
                subVideo[v - startId] = input[v](roi);

            vector<float> feat;
            hog3D.constructFeature(subVideo, feat);

            for(auto j=0; j<descriptors.cols(); ++j) {
                descriptors_.at<float>(i, j) = feat[j];
            }
        }

    }
}//namespace cv