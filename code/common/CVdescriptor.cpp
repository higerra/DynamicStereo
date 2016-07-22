//
// Created by yanhang on 7/20/16.
//

#include "CVdescriptor.h"
using namespace std;
namespace cv {
    CVHoG3D::CVHoG3D(const int ss_, const int sr_, const int M_, const int N_, const int kSubBlock_)
            : M(M_), N(N_), kSubBlock(kSubBlock_), sigma_s(ss_),  sigma_r(sr_){}

    void CVHoG3D::prepareImage(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output) const {
        CHECK(!input.empty());
        dynamic_stereo::Feature::compute3DGradient(input, output);
    }
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
            cv::Rect roi((int)pt.x - sigma_s/2, (int)pt.y-sigma_s/2, sigma_s, sigma_s);

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


    CVColor3D::CVColor3D(const int ss_, const int sr_, const int M_, const int N_)
            :sigma_s(ss_), sigma_r(sr_), M(M_), N(N_){}

    void CVColor3D::prepareImage(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output) const {
        CHECK(!input.empty());
        output.resize(input.size());
        for(auto v=0; v<input.size(); ++v){
            input[v].convertTo(output[v], CV_32F);
        }
    }
    void CVColor3D::compute(InputArray image,
                         CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                         OutputArray descriptors){
        CHECK(!keypoints.empty());
        vector<Mat> input;
        image.getMatVector(input);
        CHECK_GE(input.size(), sigma_r);

        const int kChannel = M*M*N*input[0].channels();

        descriptors.create((int)keypoints.size(), kChannel, CV_32FC1);
        Mat& descriptors_ = descriptors.getMatRef();

        dynamic_stereo::Feature::Color3D color3d(M, N);
        for(auto i=0; i<keypoints.size(); ++i){
            vector<Mat> subVideo((size_t)sigma_r);
            const Point2f& pt = keypoints[i].pt;
            cv::Rect roi((int)pt.x - sigma_s/2, (int)pt.y-sigma_s/2, sigma_s, sigma_s);

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
            color3d.constructFeature(subVideo, feat);

            for(auto j=0; j<descriptors.cols(); ++j) {
                descriptors_.at<float>(i, j) = feat[j];
            }
        }
    }
}//namespace cv