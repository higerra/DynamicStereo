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

        void extractSegmentFeature(const std::vector<cv::Mat> &images, const std::vector<ML::PixelGroup> &pixelGroups,
                                   std::vector<std::vector<float> > &feats) {
            CHECK(!images.empty());
            for (const auto &pg: pixelGroups) {
                vector<float> curRegionFeat;
                vector<float> color, shape, position;
                ML::computeColor(images, pg, color);
                ML::computeShape(pg, images[0].cols, images[0].rows, shape);
                ML::computePosition(pg, images[0].cols, images[0].rows, position);
                curRegionFeat.insert(curRegionFeat.end(), color.begin(), color.end());
                curRegionFeat.insert(curRegionFeat.end(), shape.begin(), shape.end());
                curRegionFeat.insert(curRegionFeat.end(), position.begin(), position.end());
                feats.push_back(curRegionFeat);
            }
        }

        void detectVideo(const std::vector<cv::Mat> &images,
                         cv::Ptr<cv::ml::StatModel> classifier, const cv::Mat &codebook,
                         const std::vector<float> &levelList, cv::Mat &output, const VisualWordOption &vw_option) {
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
            printf("Sample keypoints...\n");
            vector<cv::KeyPoint> keypoints;
            sampleKeyPoints(featureImages, keypoints, vw_option.sigma_s, vw_option.sigma_r);

            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
            cv::BOWImgDescriptorExtractor extractor(descriptorExtractor, matcher);
            extractor.setVocabulary(codebook);
            printf("descriptor size: %d\n", extractor.descriptorSize());

            Mat segmentVote(images[0].size(), CV_32FC1, Scalar::all(0.0f));
            for (auto level: levelList) {
                Mat segments;
                video_segment::segment_video(images, segments, level);
                vector<ML::PixelGroup> pixelGroup;
                const int kSeg = ML::regroupSegments(segments, pixelGroup);
                vector<vector<KeyPoint> > segmentKeypoints((size_t) kSeg);
                for (const auto &kpt: keypoints) {
                    int sid = segments.at<int>(kpt.pt);
                    segmentKeypoints[sid].push_back(kpt);
                }
                Mat bowFeature(kSeg, codebook.rows, CV_32FC1, Scalar::all(0));
                vector<vector<float> > regionFeature;
                extractSegmentFeature(images, pixelGroup, regionFeature);
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
                printf("Predicting...\n");
                Mat response;
                classifier->predict(featureMat, response);
                CHECK_EQ(response.rows, kSeg);

                for (auto y = 0; y < segmentVote.rows; ++y) {
                    for (auto x = 0; x < segmentVote.cols; ++x) {
                        int sid = segments.at<int>(y, x);
                        if (response.at<float>(sid, 0) > 0.5)
                            segmentVote.at<float>(y, x) += 1.0f;
                    }
                }
            }

            output.create(segmentVote.size(), CV_8UC1);
            output.setTo(Scalar::all(0));

            for (auto y = 0; y < segmentVote.rows; ++y) {
                for (auto x = 0; x < segmentVote.cols; ++x) {
                    if (segmentVote.at<float>(y, x) > (float) levelList.size() / 2)
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