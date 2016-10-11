//
// Created by yanhang on 7/27/16.
//

#include "../external/segment_gb/segment-image.h"

#include "videosegmentation.h"
#include "pixel_feature.h"

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

        int segment_video(const std::vector<cv::Mat> &input, cv::Mat &output, const VideoSegmentOption& option) {
            CHECK(!input.empty());
            const int width = input[0].cols;
            const int height = input[0].rows;

            std::vector<cv::Mat> smoothed(input.size());
            for (auto v = 0; v < input.size(); ++v) {
                cv::blur(input[v], smoothed[v], cv::Size(option.smooth_size, option.smooth_size));
            }

            cv::Mat edgeMap;
            edgeAggregation(smoothed, edgeMap);

            //std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor(new PixelValue());
            std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor;

            if (option.pixel_feture_type == PixelFeature::PIXEL_VALUE)
                pixel_extractor.reset(new PixelValue());
            else if (option.pixel_feture_type == PixelFeature::BRIEF)
                pixel_extractor.reset(new BRIEFWrapper());
            else
                CHECK(true) << "Unsupported pixel feature type";

            std::shared_ptr<TemporalFeatureExtractorBase> temporal_extractor;

            if (option.temporal_feature_type == TemporalFeature::TRANSITION_PATTERN) {
                temporal_extractor.reset(new TransitionPattern(pixel_extractor.get(), option.stride1, option.stride2, option.theta));
            }else if (option.temporal_feature_type == TemporalFeature::TRANSITION_COUNTING) {
                temporal_extractor.reset(new TransitionCounting(pixel_extractor.get(), option.stride1, option.stride2, option.theta));
            }else if(option.temporal_feature_type == TemporalFeature::TRANSITION_AND_APPEARANCE) {
                constexpr double w_appearance = 0.02;
                constexpr double w_transition = 1 - w_appearance;
                temporal_extractor.reset(new TransitionAndAppearance(pixel_extractor.get(), pixel_extractor.get(),
                                                                     option.stride1, option.stride2, option.theta,
                                                                     w_transition, w_appearance));
            }else{
                CHECK(true) << "Unsupported temporal feature type";
            }

            const DistanceMetricBase *feature_comparator = CHECK_NOTNULL(temporal_extractor->getDefaultComparator());

            vector<cv::Mat> pixelFeatures(smoothed.size());
#pragma omp parallel for
            for (auto v = 0; v < smoothed.size(); ++v) {
                pixel_extractor->extractAll(smoothed[v], pixelFeatures[v]);
            }

            {
                //Debug: temporal color histogram
                const int tp1 = 267 * width + 1198;
                const int tp2 = 267 * width + 1206;

                Mat histogram;
                ColorHistogram hist_extractor(ColorHistogram::HSV, {4,4,4}, width, height, 1);
                hist_extractor.computeFromPixelFeature(pixelFeatures, histogram);
                printf("Histogram:\n");
                cout << histogram.row(tp1) << endl;
                cout << histogram.row(tp2) << endl;

                //Debug: Average
                Mat average;
                TemporalAverage average_extractor;
                average_extractor.computeFromPixelFeature(pixelFeatures, average);
                printf("Average:\n");
                cout << average.row(tp1) << endl;
                cout << average.row(tp2) << endl;

            }

            Mat featuresMat;
            if(option.temporal_feature_type == TemporalFeature::TRANSITION_AND_APPEARANCE){
                dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)
                        ->computeFromPixelAndAppearanceFeature(pixelFeatures, pixelFeatures, featuresMat);
            }else {
                temporal_extractor->computeFromPixelFeature(pixelFeatures, featuresMat);
            }

            CHECK_EQ(featuresMat.rows, width * height);
#if false
            {
                //debug, inspect some of the feature
                const int tx = 249, ty = 210;
                if (tx >= 0 && ty >= 0) {
                    dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(
                            featuresMat.row(ty * width + tx));
                }
            }
#endif
            // build graph
            std::vector<edge> edges;
            edges.reserve((size_t) width * height);

            constexpr int dx = -1;
            constexpr int dy = -1;

//            printf("Pixel feature for (%d,%d)\n", dx+1, dy);
//            for(auto v=0; v<pixelFeatures.size(); ++v){
//                cout << pixelFeatures[v].row(dy*width+dx+1) << endl;
//            }

            //8 neighbor
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float edgeness = edgeMap.at<float>(y, x);
                    bool verbose = (x == dx) && (y == dy);
                    if(verbose){
                        printf("====================\n");
                        printf("Debug info for (%d,%d)\n", x,y);
                        printf("descriptor of (%d,%d)\t0.000\t", x, y);
                        dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(featuresMat.row(y*width+x));
                   }
                    if (x < width - 1) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = y * width + (x + 1);
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);

                        if(verbose) {
                            printf("descriptor of (%d,%d)\t%.3f\t", curEdge.b%width, curEdge.b/width, curEdge.w);
                            dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(featuresMat.row(curEdge.b));
                        }
                    }

                    if (y < height - 1) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);

                        if(verbose) {
                            printf("descriptor of (%d,%d)\t%.3f\t", curEdge.b%width, curEdge.b/width, curEdge.w);
                            dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(featuresMat.row(curEdge.b));
                        }
                    }

                    if ((x < width - 1) && (y < height - 1)) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);

                        if(verbose) {
                            printf("descriptor of (%d,%d)\t%.3f\t", curEdge.b%width, curEdge.b/width, curEdge.w);
                            dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(featuresMat.row(curEdge.b));
                        }

                    }

                    if ((x < width - 1) && (y > 0)) {
                        edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y - 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);

                        if(verbose) {
                            printf("descriptor of (%d,%d)\t%.3f\t", curEdge.b%width, curEdge.b/width, curEdge.w);
                            dynamic_pointer_cast<TransitionAndAppearance>(temporal_extractor)->printFeature(featuresMat.row(curEdge.b));
                        }

                    }
                }
            }

            std::unique_ptr<universe> u(segment_graph(width * height, edges, option.threshold));
            // post process small components
            for (const auto &e: edges) {
                int a = u->find(e.a);
                int b = u->find(e.b);
                if ((a != b) && ((u->size(a) < option.min_size) || (u->size(b) < option.min_size))) {
                    u->join(a, b);
                }
            }
            output = cv::Mat(height, width, CV_32SC1, cv::Scalar::all(0));
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

            if(option.refine) {
                LOG(INFO) << "Running refinement...";
                mfGrabCut(input, output, 1, 1);
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

                for(auto y=0; y<height; ++y){
                    for(auto x=0; x<width; ++x){
                        if(pLabel[y*width+x] == l)
                            resultMask.at<uchar>(y,x) = (uchar)255;
                    }
                }
            }

            Mat result;
            cv::connectedComponents(resultMask, result);
            return result;
        }
    }//video_segment
}//namespace dynamic_stereo
