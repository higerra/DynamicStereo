//
// Created by yanhang on 7/27/16.
//

#include "videosegmentation.h"
#include "../external/segment_gb/segment-image.h"

using namespace std;
using namespace cv;
using namespace segment_gb;

namespace dynamic_stereo {

    namespace video_segment {

        void edgeAggregation(const VideoMat &input, cv::Mat &output) {
            CHECK(!input.empty());
            output.create(input[0].size(), CV_32FC1);
            output.setTo(cv::Scalar::all(0));
            for (auto i = 0; i < input.size(); ++i) {
                cv::Mat edge_sobel(input[i].size(), CV_32FC1, cv::Scalar::all(0));
                cv::Mat gray, gx, gy;
                cvtColor(input[i], gray, CV_BGR2GRAY);
                cv::Sobel(gray, gx, CV_32F, 1, 0);
                cv::Sobel(gray, gy, CV_32F, 0, 1);
                for (auto y = 0; y < gray.rows; ++y) {
                    for (auto x = 0; x < gray.cols; ++x) {
                        float ix = gx.at<float>(y, x);
                        float iy = gy.at<float>(y, x);
                        edge_sobel.at<float>(y, x) = std::sqrt(ix * ix + iy * iy + FLT_EPSILON);
                    }
                }
                output += edge_sobel;
            }

            double maxedge, minedge;
            cv::minMaxLoc(output, &minedge, &maxedge);
            if (maxedge > 0)
                output /= maxedge;
        }

        int segment_video(const std::vector<cv::Mat> &input, cv::Mat &output,
                          const int smoothSize, const float c, const float theta, const int min_size,
                          const PixelFeature pfType,
                          const TemporalFeature tfType) {
            CHECK(!input.empty());
            const int width = input[0].cols;
            const int height = input[0].rows;

            std::vector<cv::Mat> smoothed(input.size());
            for (auto v = 0; v < input.size(); ++v) {
                cv::blur(input[v], smoothed[v], cv::Size(smoothSize, smoothSize));
            }

            cv::Mat edgeMap;
            edgeAggregation(smoothed, edgeMap);

            const int stride1 = 8;
            const int stride2 = (int) input.size() / 2;

            //std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor(new PixelValue());
            std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor;

            if (pfType == PixelFeature::PIXEL)
                pixel_extractor.reset(new PixelValue());
            else if (pfType == PixelFeature::BRIEF)
                pixel_extractor.reset(new BRIEFWrapper());
            else
                CHECK(true) << "Unsupported pixel feature type";

            std::shared_ptr<TemporalFeatureExtractorBase> temporal_extractor;

            if (tfType == TemporalFeature::TRANSITION_PATTERN)
                temporal_extractor.reset(new TransitionPattern(pixel_extractor.get(), stride1, stride2, theta));
            else if (tfType == TemporalFeature::TRANSITION_COUNTING)
                temporal_extractor.reset(new TransitionCounting(pixel_extractor.get(), stride1, stride2, theta));
            else
                CHECK(true) << "Unsupported temporal feature type";

            const DistanceMetricBase *feature_comparator = temporal_extractor->getDefaultComparator();

            vector<cv::Mat> pixelFeatures(smoothed.size());
#pragma omp parallel for
            for (auto v = 0; v < smoothed.size(); ++v) {
                pixel_extractor->extractAll(smoothed[v], pixelFeatures[v]);
            }

            Mat featuresMat;
            //a working around: receive Mat type from the function and fit it to vector<vector< > >
            temporal_extractor->computeFromPixelFeature(pixelFeatures, featuresMat);

            {
                //debug, inspect some of the feature
                const int tx = -1, ty = -1;
                if (tx >= 0 && ty >= 0) {
                    dynamic_pointer_cast<TransitionPattern>(temporal_extractor)->printFeature(
                            featuresMat.row(ty * width + tx));
                }
            }

            // build graph
            std::vector<edge> edges;
            edges.reserve((size_t) width * height);
            //8 neighbor
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float edgeness = edgeMap.at<float>(y, x);
                    if (x < width - 1) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = y * width + (x + 1);
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if (y < height - 1) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if ((x < width - 1) && (y < height - 1)) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if ((x < width - 1) && (y > 0)) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y - 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }
                }
            }

            std::unique_ptr<universe> u(segment_graph(width * height, edges, c));
            // post process small components
            for (const auto &e: edges) {
                int a = u->find(e.a);
                int b = u->find(e.b);
                if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size))) {
                    u->join(a, b);
                }
            }

            output = cv::Mat(height, width, CV_32S, cv::Scalar::all(0));
            int *pOutput = (int *) output.data;
            //remap labels
            vector<int> labelMap((size_t) width * height, -1);
            int nLabel = -1;
            for (auto i = 0; i < width * height; ++i) {
                int comp = u->find(i);
                if (labelMap[comp] < 0)
                    labelMap[comp] = ++nLabel;
            }
            nLabel++;

            for (auto i = 0; i < width * height; ++i) {
                int comp = u->find(i);
                pOutput[i] = labelMap[comp];
            }

            return nLabel;
        }

        cv::Mat visualizeSegmentation(const cv::Mat &input) {
            return segment_gb::visualizeSegmentation(input);
        }

    }//video_segment
}//namespace dynamic_stereo