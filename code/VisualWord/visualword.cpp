//
// Created by yanhang on 7/19/16.
//

#include "visualword.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo {
    namespace VisualWord {
        void sampleKeyPoints(const std::vector<cv::Mat> &input, std::vector<cv::KeyPoint> &keypoints, const int sigma_s,
                             const int sigma_r) {
            CHECK(!input.empty());
            const int width = input[0].cols;
            const int height = input[0].rows;
            const int kFrame = (int) input.size();

            const int &rS = sigma_s;
            const int &rT = sigma_r;
            keypoints.reserve((size_t) (width / rS * height / rS * kFrame / rT));

            for (auto x = rS + 1; x < width - rS; x += rS) {
                for (auto y = rS + 1; y < height - rS; y += rS) {
                    for (auto t = rT + 1; t < kFrame - rT; t += rT) {
                        cv::KeyPoint keypt;
                        keypt.pt = cv::Point2f(x, y);
                        keypt.octave = t;
                        keypoints.push_back(keypt);
                    }
                }
            }
        }


        void detectVideo(const std::vector<cv::Mat> &images,
                         cv::Ptr<cv::ml::StatModel> classifier, const cv::Mat &codebook,
                         const std::vector<float> &levelList, cv::Mat &output, const VisualWordOption &vw_option,
                         cv::InputArrayOfArrays inputSegments, cv::OutputArrayOfArrays rawSegments) {
            CHECK(!images.empty());
            CHECK(classifier.get());
            CHECK(codebook.data);
            CHECK(!levelList.empty());
            cv::Ptr<cv::Feature2D> descriptorExtractor;
            if (vw_option.pixDesc == HOG3D)
                descriptorExtractor.reset(new CVHoG3D(vw_option.sigma_s, vw_option.sigma_r));
            else if (vw_option.pixDesc == COLOR3D)
                descriptorExtractor.reset(new CVColor3D(vw_option.sigma_s, vw_option.sigma_r));

            vector<Mat> featureImages;
            descriptorExtractor.dynamicCast<CV3DDescriptor>()->prepareImage(images, featureImages);
            vector<cv::KeyPoint> keypoints;
            sampleKeyPoints(featureImages, keypoints, vw_option.sigma_s, vw_option.sigma_r);

            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
            cv::BOWImgDescriptorExtractor extractor(descriptorExtractor, matcher);
            extractor.setVocabulary(codebook);
            char buffer[128] = {};

            if (rawSegments.needed()) {
                rawSegments.create((int) levelList.size(), 1, CV_32S);
                for (int i = 0; i < levelList.size(); ++i)
                    rawSegments.create(images[0].size(), CV_32S, i);
            }

            vector<Mat> segments;
            if (!inputSegments.empty()) {
                inputSegments.getMatVector(segments);
                CHECK_EQ(segments.size(), levelList.size());
            }


            vector<Mat> classification_level(levelList.size());

            /*!
             * Note: There are some memory inefficient operations inside region descriptor, so there won't be enough
             * memory for parallel execution.
             */
//#pragma omp parallel for
            for (int lid=0; lid<levelList.size(); ++lid) {
                Mat segment;
                if (!segments.empty()) {
                    segment = segments[lid];
                    CHECK_EQ(segment.size(), images[0].size());
                } else {
                    video_segment::VideoSegmentOption option(levelList[lid]);
                    option.refine = false;
                    option.temporal_feature_type = video_segment::TRANSITION_AND_APPEARANCE;
                    LOG(INFO) << "Segmenting at " << levelList[lid];
                    video_segment::segment_video(images, segment, option);
                }

                if (rawSegments.needed())
                    segment.copyTo(rawSegments.getMat(lid));

                vector<ML::PixelGroup> pixelGroup;
                const int kSeg = ML::regroupSegments(segment, pixelGroup);
                vector<vector<KeyPoint> > segmentKeypoints((size_t) kSeg);
                for (const auto &kpt: keypoints) {
                    int sid = segment.at<int>(kpt.pt);
                    segmentKeypoints[sid].push_back(kpt);
                }

                Mat bowFeature(kSeg, codebook.rows, CV_32FC1, Scalar::all(0));
                vector<vector<float> > regionFeature;
                ML::extractSegmentFeature(images, pixelGroup, regionFeature);

                for (auto sid = 0; sid < kSeg; ++sid) {
                    if (!segmentKeypoints[sid].empty()) {
                        Mat bow;
                        extractor.compute(featureImages, segmentKeypoints[sid], bow);
                        bow.copyTo(bowFeature.rowRange(sid, sid + 1));
                    }
                }

                Mat featureMat(kSeg, codebook.rows + (int) regionFeature[0].size(), CV_32FC1, Scalar::all(0));
                bowFeature.copyTo(featureMat.colRange(0, codebook.rows));
                for (auto sid = 0; sid < kSeg; ++sid) {
                    for (auto j = 0; j < regionFeature[sid].size(); ++j)
                        featureMat.at<float>(sid, j + codebook.rows) = regionFeature[sid][j];
                }

                Mat response;
                classifier->predict(featureMat, response);
                CHECK_EQ(response.rows, kSeg);

                classification_level[lid].create(images[0].size(), CV_32FC1);
                classification_level[lid].setTo(Scalar::all(0));
                for (auto y = 0; y < classification_level[lid].rows; ++y) {
                    for (auto x = 0; x < classification_level[lid].cols; ++x) {
                        int sid = segment.at<int>(y, x);
                        if (response.at<float>(sid, 0) > 0.5)
                            classification_level[lid].at<float>(y, x) += 1.0f;
                    }
                }
            }

            Mat segmentVote(images[0].size(), CV_32FC1, Scalar::all(0.0f));
            for(const auto& m: classification_level){
                segmentVote += m;
            }

            output.create(segmentVote.size(), CV_8UC1);
            output.setTo(Scalar::all(0));

            for (auto y = 0; y < segmentVote.rows; ++y) {
                for (auto x = 0; x < segmentVote.cols; ++x) {
                    if (segmentVote.at<float>(y, x) > (float) levelList.size() / 3)
                        output.at<uchar>(y, x) = (uchar) 255;
                }
            }
        }

        double testClassifier(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::StatModel> classifier) {
            CHECK(testPtr.get());
            CHECK(classifier.get());

            Mat result;
            classifier->predict(testPtr->getSamples(), result);
            Mat groundTruth;
            testPtr->getResponses().convertTo(groundTruth, CV_32F);

            CHECK_EQ(groundTruth.rows, result.rows);
            float acc = 0.0f;
            for (auto i = 0; i < result.rows; ++i) {
                float gt = groundTruth.at<float>(i, 0);
                float res = result.at<float>(i, 0);
                if (std::abs(gt - res) <= 0.1)
                    acc += 1.0f;
            }
            return acc / (float) result.rows;
        }
    }//namespace VisualWord
}//namespace dynamic_stereo