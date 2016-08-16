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
                          const float c, const bool refine, const int smoothSize, const float theta, const int min_size,
                          const int stride1, const int stride2,
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
            int nLabel = 0;
            for (auto i = 0; i < width * height; ++i) {
                int comp = u->find(i);
                if (labelMap[comp] < 0)
                    labelMap[comp] = nLabel++;
            }

            for (auto i = 0; i < width * height; ++i) {
                int comp = u->find(i);
                pOutput[i] = labelMap[comp];
            }

            if(refine) {
                printf("Running refinement...\n");
                mfGrabCut(input, output, 1);
            }
            nLabel = compressSegment(output);
            return nLabel;
        }

        cv::Mat visualizeSegmentation(const cv::Mat &input) {
            return segment_gb::visualizeSegmentation(input);
        }

        int compressSegment(cv::Mat& segment){
            CHECK(segment.data);
            CHECK_EQ(segment.type(), CV_32SC1);
            double minL, maxL;
            cv::minMaxLoc(segment, &minL, &maxL);
            int kSeg = (int)maxL + 1;
            vector<int> labelMap((size_t)kSeg, -1);
            int index = 0;
            for(auto y=0; y<segment.rows; ++y){
                for(auto x=0; x<segment.cols; ++x){
                    const int lid = segment.at<int>(y,x);
                    if(labelMap[lid] < 0)
                        labelMap[lid] = index++;
                }
            }
            for(auto y=0; y<segment.rows; ++y){
                for(auto x=0; x<segment.cols; ++x){
                    const int lid = segment.at<int>(y,x);
                    segment.at<int>(y,x) = labelMap[lid];
                }
            }
            return index;
        }

        Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask){
            CHECK(!images.empty());
            const int width = images[0].cols;
            const int height = images[0].rows;

            Mat resultMask(height, width, CV_8UC1, Scalar::all(0));

            Mat labels, stats, centroid;
            int nLabel = cv::connectedComponentsWithStats(mask, labels, stats, centroid);
            const int* pLabel = (int*) labels.data;

            const int min_area = 50;
            const double maxRatioOcclu = 0.3;

            int kOutputLabel = 1;

            const int testL = -1;

            const int localMargin = std::min(width, height) / 10;
            for(auto l=1; l<nLabel; ++l){
                if(testL > 0 && l != testL)
                    continue;

                const int area = stats.at<int>(l, CC_STAT_AREA);
                //search for bounding box.
                const int cx = stats.at<int>(l,CC_STAT_LEFT) + stats.at<int>(l,CC_STAT_WIDTH) / 2;
                const int cy = stats.at<int>(l,CC_STAT_TOP) + stats.at<int>(l,CC_STAT_HEIGHT) / 2;

                printf("========================\n");
                printf("label:%d/%d, centroid:(%d,%d), area:%d\n", l, nLabel, cx, cy, area);
                if(area < min_area) {
                    printf("Area too small\n");
                    continue;
                }

                //filter out mostly occlued areas
                int nOcclu = 0;
                for(auto y=0; y<height; ++y){
                    for(auto x=0; x<width; ++x){
                        if(pLabel[y*width+x] != l)
                            continue;
                        int pixOcclu = 0;
                        for(auto v=0; v<images.size(); ++v){
                            if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
                                pixOcclu++;
                        }
                        if(pixOcclu > (int)images.size() / 3)
                            nOcclu++;
                    }
                }
                if(nOcclu > maxRatioOcclu * area) {
                    printf("Violate occlusion constraint\n");
                    continue;
                }
                kOutputLabel++;
            }

            Mat result;
            cv::connectedComponents(resultMask, result);
            return result;
        }
    }//video_segment
}//namespace dynamic_stereo
