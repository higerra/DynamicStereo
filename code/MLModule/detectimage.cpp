//
// Created by yanhang on 8/4/16.
//

#include "detectimage.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{
    namespace ML {
        void detectImage(const std::vector<cv::Mat> &images, const std::vector<cv::Mat>& segmentation,
                         cv::Ptr<cv::ml::StatModel> classifier, cv::Mat &output) {
            CHECK(!images.empty());
            CHECK(!segmentation.empty());
            CHECK_EQ(images[0].size(), segmentation[0].size());
            vector<Mat> gradient(images.size());
            for (auto i = 0; i < images.size(); ++i) {
                MLUtility::computeGradient(images[i], gradient[i]);
                images[i].convertTo(images[i], CV_32FC3);
            }

            //empty ground truth
            Mat gt;
            Mat segmentVote(images[0].size(), CV_32FC1, Scalar::all(0.0f));
            printf("Running classification...\n");

            for (auto seg: segmentation) {
                double minL, maxL;
                cv::minMaxLoc(seg, &minL, &maxL);
                const int kSeg = (int) maxL + 1;

                TrainSet testset;
                printf("Extracting feature...\n");
                extractFeature(images, gradient, seg, gt, testset);


                cv::Ptr<ml::TrainData> testPtr = MLUtility::convertTrainData(testset);
                CHECK(testPtr.get());
                Mat result;
                classifier->predict(testPtr->getSamples(), result);
                CHECK_EQ(result.rows, testset[0].size());
                vector<bool> segmentLabel((size_t) kSeg, false);
                for (auto i = 0; i < result.rows; ++i) {
                    if (result.at<float>(i, 0) > 0.5) {
                        int sid = testset[0][i].id;
                        segmentLabel[sid] = true;
                    }
                }
                for (auto y = 0; y < segmentVote.rows; ++y) {
                    for (auto x = 0; x < segmentVote.cols; ++x) {
                        int sid = seg.at<int>(y, x);
                        if (segmentLabel[sid]) {
                            segmentVote.at<float>(y, x) += 1.0f;
                        }
                    }
                }
            }//for(auto levelList)

            output.create(segmentVote.size(), CV_8UC1);
            output.setTo(Scalar::all(0));
            for (auto y = 0; y < output.rows; ++y) {
                for (auto x = 0; x < output.cols; ++x) {
                    if (segmentVote.at<float>(y, x) > (float) segmentation.size() / 2)
                        output.at<uchar>(y, x) = (uchar) 255;
                }
            }
        }
    }//namespace ML
}//namespace dynamic_stereo