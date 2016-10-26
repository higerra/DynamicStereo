//
// Created by yanhang on 7/27/16.
//

#include "videosegmentation.h"
#include "pixel_feature.h"
#include "region_feature.h"

using namespace std;
using namespace cv;

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

            char buffer[128] = {};
            std::vector<cv::Mat> smoothed(input.size());
            for (auto v = 0; v < input.size(); ++v) {
                cv::blur(input[v], smoothed[v], cv::Size(option.smooth_size, option.smooth_size));
            }

            cv::Mat edgeMap;
            edgeAggregation(smoothed, edgeMap);
//            imshow("Edge Map", edgeMap);
//            waitKey(0);

            //std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor(new PixelValue());
            std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor;

            if (option.pixel_feture_type == PixelFeature::PIXEL_VALUE)
                pixel_extractor.reset(new PixelValue());
            else if (option.pixel_feture_type == PixelFeature::BRIEF)
                pixel_extractor.reset(new BRIEFWrapper());
            else
                CHECK(true) << "Unsupported pixel feature type";

            std::shared_ptr<TemporalFeatureExtractorBase> temporal_extractor;

            LOG(INFO) << "Creating extractor";
            if (option.temporal_feature_type == TemporalFeature::TRANSITION_PATTERN) {
                temporal_extractor.reset(new TransitionPattern(input.size(), option.stride1, option.stride2, option.theta,
                                                               pixel_extractor->getDefaultComparator()));
            }else if(option.temporal_feature_type == TemporalFeature::COMBINED) {
                const vector<int> kBins{8,8,8};
                constexpr int R = 0;
                std::vector<std::shared_ptr<TemporalFeatureExtractorBase> > component_extractors(2);
                component_extractors[0].reset(new ColorHistogram(ColorHistogram::LAB, kBins, width, height, R));
                //component_extractors[0].reset(new TemporalAverage());
                component_extractors[1].reset(new TransitionPattern(input.size(), option.stride1, option.stride2, option.theta,
                                                                    pixel_extractor->getDefaultComparator()));
                temporal_extractor.reset(new CombinedTemporalFeature(component_extractors, {option.w_appearance, option.w_transition}));

            }else{
                CHECK(true) << "Unsupported temporal feature type";
            }

            LOG(INFO) << "Feature dimension: " << temporal_extractor->getDim();

            const DistanceMetricBase *feature_comparator = CHECK_NOTNULL(temporal_extractor->getDefaultComparator());

            LOG(INFO) << "Extracting pixel features";
            vector<cv::Mat> pixelFeatures(smoothed.size());
#pragma omp parallel for
            for (auto v = 0; v < smoothed.size(); ++v) {
                pixel_extractor->extractAll(smoothed[v], pixelFeatures[v]);
            }

            LOG(INFO) << "Extracting temporal features";
            Mat featuresMat;
            if(option.temporal_feature_type == TemporalFeature::COMBINED){
                vector<vector<Mat> > component_pixel_features{pixelFeatures, pixelFeatures};
                dynamic_pointer_cast<CombinedTemporalFeature>(temporal_extractor)->
                        computeFromPixelFeatures(component_pixel_features, featuresMat);
            }else {
                temporal_extractor->computeFromPixelFeature(pixelFeatures, featuresMat);
            }

            CHECK_EQ(featuresMat.rows, width * height);
#if true
            {
                DistanceCorrelation dis_cor;
                //debug, inspect some of the feature
                const int dx1 = 1007, dy1 = 200, dx2 = 1007, dy2 = 199;
                printf("(%d,%d):\n", dx1, dy1);
                temporal_extractor->printFeature(featuresMat.row(dy1*width+dx1));
                printf("(%d,%d):\n", dx2, dy2);
                temporal_extractor->printFeature(featuresMat.row(dy2*width+dx2));
                const double edgeness = edgeMap.at<float>(dy2, dx2);
                const double appearance = dis_cor.evaluate(featuresMat.row(dy1*width+dx1).colRange(0,24),
                                                           featuresMat.row(dy2*width+dx2).colRange(0,24));
                const double raw_dis = feature_comparator->evaluate(featuresMat.row(dy1*width+dx1), featuresMat.row(dy2*width+dx2));
                printf("Distance: edge: %.5f, raw: %.5f, overall: %.5f\n", edgeness, raw_dis, edgeness * raw_dis);
                printf("Appearance distance: %.5f\n", appearance);
            }
#endif
            // build graph
            std::vector<segment_gb::edge> edges;
            edges.reserve((size_t) width * height);

            LOG(INFO) << "Segmenting";
            constexpr float base_edgeness = 0;
            //8 neighbor
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    //float edgeness = edgeMap.at<float>(y, x);
                    float edgeness = base_edgeness + edgeMap.at<float>(y, x) * (1 - base_edgeness);
                    //float edgeness = 1.0;
                    if (x < width - 1) {
                        segment_gb::edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = y * width + (x + 1);
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if (y < height - 1) {
                        segment_gb::edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if ((x < width - 1) && (y < height - 1)) {
                        segment_gb::edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y + 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }

                    if ((x < width - 1) && (y > 0)) {
                        segment_gb::edge curEdge;
                        curEdge.a = y * width + x;
                        curEdge.b = (y - 1) * width + x + 1;
                        curEdge.w = feature_comparator->evaluate(featuresMat.row(curEdge.a),
                                                                 featuresMat.row(curEdge.b)) * edgeness;
                        edges.push_back(curEdge);
                    }
                }
            }

            std::unique_ptr<segment_gb::universe> u(segment_gb::segment_graph(width * height, edges, option.threshold));
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

        void BuildEdgeMap(const std::vector<Region*>& regions, std::vector<segment_gb::edge> & edge_map,
                                 const int width, const int height) {
            Mat pixel_region(height, width, CV_32SC1, Scalar::all(0));
            for (auto rid = 0; rid < regions.size(); ++rid) {
                CHECK(regions[rid] != nullptr);
                for (auto pid: regions[rid]->pix_id) {
                    pixel_region.at<int>(pid / width, pid % width) = rid;
                }
            }
            vector<vector<int> > intersect_count(regions.size());
            for (auto &ic: intersect_count) {
                ic.resize(regions.size(), 0);
            }

            for (auto y = 0; y < height; ++y) {
                for (auto x = 0; x < width; ++x) {
                    int rid1 = pixel_region.at<int>(y, x);
                    int rid2;

                    if (x < width - 1) {
                        rid2 = pixel_region.at<int>(y, x + 1);
                        if (rid1 != rid2) {
                            intersect_count[rid1][rid2] += 1;
                        }
                    }

                    if (x < width - 1 && y < height - 1) {
                        rid2 = pixel_region.at<int>(y + 1, x + 1);
                        if (rid1 != rid2) {
                            intersect_count[rid1][rid2] += 1;
                        }
                    }
                    if (y < height - 1) {
                        rid2 = pixel_region.at<int>(y + 1, x);
                        if (rid1 != rid2) {
                            intersect_count[rid1][rid2] += 1;
                        }
                    }
                    if (x < width - 1 && y > 0) {
                        rid2 = pixel_region.at<int>(y - 1, x + 1);
                        if (rid1 != rid2) {
                            intersect_count[rid1][rid2] += 1;
                        }
                    }
                }
            }

            for (auto rid1 = 0; rid1 < regions.size() - 1; ++rid1) {
                for (auto rid2 = rid1 + 1; rid2 < regions.size(); ++rid2) {
                    if (intersect_count[rid1][rid2] > 0) {
                        segment_gb::edge new_edge;
                        new_edge.a = rid1;
                        new_edge.b = rid2;
                        new_edge.w = (float) intersect_count[rid1][rid2];
                        edge_map.push_back(new_edge);
                    }
                }
            }
        }

        int HierarchicalSegmentation(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const VideoSegmentOption& option){
            CHECK(!input.empty());
            //first perform dense segment
            const int width = input[0].cols;
            const int height = input[0].rows;

            //Initialization
            LOG(INFO) << "Initializing region segmentation";
            std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor;
            std::shared_ptr<RegionFeatureExtractorBase> region_temporal_extractor;

            pixel_extractor.reset(new PixelValue());
            TemporalAverage region_average;

            if(option.region_temporal_feature_type == TemporalFeature::TRANSITION_PATTERN){
                CHECK(true) << "Not implemented";
            }else if(option.region_temporal_feature_type == TemporalFeature::COMBINED) {
                const vector<int> kBins{8, 8, 8};
                const ColorHistogram::ColorSpace cspace = ColorHistogram::LAB;
                vector<shared_ptr<RegionFeatureExtractorBase> > region_extractors(2);
                region_extractors[0].reset(new RegionColorHist(cspace, kBins, width, height));
                region_extractors[1].reset(
                        new RegionTransitionPattern((int) input.size(), option.stride1, option.stride2,
                                                    option.theta, pixel_extractor->getDefaultComparator(),
                                                    &region_average));
                region_temporal_extractor.reset(
                        new RegionCombinedFeature(region_extractors, {option.w_appearance, option.w_transition}));
            }

            const DistanceMetricBase* temporal_comparator = region_temporal_extractor->getDefaultComparator();

            VideoSegmentOption dense_option = option;
            dense_option.threshold = 0.2;

            LOG(INFO) << "Computing dense segment";
            Mat iter_segment;
            int num_segments = segment_video(input, iter_segment, dense_option);
            LOG(INFO) << "Number of dense segments: " << num_segments;
            CHECK_EQ(iter_segment.cols, width);
            CHECK_EQ(iter_segment.rows, height);
            CHECK_EQ(iter_segment.type(), CV_32SC1);
            int* iter_segment_ptr = (int*) iter_segment.data;
            LOG(INFO) << "Extracting pixel features";
            vector<Mat> pixel_features(input.size());

#pragma omp parallel for
            for(auto v=0; v<input.size(); ++v){
                pixel_extractor->extractAll(input[v], pixel_features[v]);
            }

            float hier_min_size = (float)option.min_size;
            float hier_threshold = dense_option.threshold;

            auto get_center = [&](const std::vector<int>& pix_id){
                Vec2d center(0,0);
                for(auto pid: pix_id){
                    center[0] += pid % width;
                    center[1] += pid / width;
                }
                center /= (double)pix_id.size();
                return center;
            };

            char buffer[128] = {};
            for(auto iter = 0; iter < option.hier_iter || option.hier_iter < 0; ++iter){
                LOG(INFO) << "------------------------------------";
                LOG(INFO) << "Hierarhical segmentation, iter " << iter;
                //construct regions
                vector<Region> regions((size_t) num_segments);
                for(auto i=0; i<width * height; ++i){
                    regions[iter_segment_ptr[i]].pix_id.push_back(i);
                }

                if(regions.size() < 2){
                    LOG(WARNING) << "Only one segment remains, break";
                    break;
                }
                vector<Region*> region_ptr(regions.size());
                for(auto rid=0; rid < regions.size(); ++rid){
                    region_ptr[rid] = &regions[rid];
                }

                LOG(INFO) << "Extracting region features";
                Mat region_features;
                if(option.region_temporal_feature_type == TemporalFeature::COMBINED){
                    dynamic_pointer_cast<RegionCombinedFeature>(region_temporal_extractor)
                            ->ExtractFromPixelFeatureArray({pixel_features, pixel_features}, region_ptr, region_features);
                }else{
                    CHECK(true)  << "Not implemented";
                }
                CHECK_EQ(region_features.rows, region_ptr.size());

                LOG(INFO) << "Building edge maps";
                vector<segment_gb::edge> edge_map;
                BuildEdgeMap(region_ptr, edge_map, width, height);

                CHECK(!edge_map.empty());
                for(auto i=0; i<edge_map.size(); ++i){
                    edge_map[i].w = temporal_comparator->
                            evaluate(region_features.row(edge_map[i].a), region_features.row(edge_map[i].b));
                }

                //update min_size and threshold
                hier_min_size = hier_min_size * 1.5;
                hier_threshold *= 1.5;

                LOG(INFO) << "Running grph segmentation";
                std::unique_ptr<segment_gb::universe> graph(segment_gb::segment_graph((int)region_ptr.size(), edge_map, hier_threshold));

                //Notice: the size of segments can not be quried by graph->size(). We need to compute the actual size of
                //segment in pixels
                vector<int> segment_sizes(region_ptr.size(), 0);
                for(auto rid=0; rid < region_ptr.size(); ++rid){
                    int sid = graph->find(rid);
                    segment_sizes[sid] += region_ptr[rid]->pix_id.size();
                }
                for(const auto& e: edge_map){
                    int sid_a = graph->find(e.a);
                    int sid_b = graph->find(e.b);
                    if ((sid_a != sid_b) && ((segment_sizes[sid_a] < (int)hier_min_size) || (segment_sizes[sid_b] < (float)hier_min_size))) {
                        segment_sizes[sid_a] += segment_sizes[sid_b];
                        segment_sizes[sid_b] = segment_sizes[sid_a];
                        graph->join(sid_a, sid_b);
                    }
                }

                iter_segment.setTo(cv::Scalar::all(-1));
                for(auto rid=0; rid < region_ptr.size(); ++rid){
                    for(auto pid: region_ptr[rid]->pix_id){
                        iter_segment_ptr[pid] = graph->find(rid);
                    }
                }

                //sanity check
                for(auto i=0; i<width * height; ++i){
                    CHECK_GE(iter_segment_ptr[i], 0);
                }
                num_segments = compressSegment(iter_segment);
                output.push_back(iter_segment.clone());
                LOG(INFO) << "Iteration " << iter << " done, number of segments: " << num_segments;
            }
            return num_segments;
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


            //morphological operation
            const cv::Size erode_R(5,5);
            const cv::Size dilate_R(7,7);

            Mat eroded, dilated;
            cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, erode_R));
            cv::dilate(eroded, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, dilate_R));

            Mat labels, stats, centroid;
            int nLabel = cv::connectedComponentsWithStats(dilated, labels, stats, centroid);
            const int* pLabel = (int*) labels.data;

            const int min_area = 50;
            const int max_area = width * height / 8;

            const double maxRatioOcclu = 0.3;

            int kOutputLabel = 1;

            const int testL = -1;

            const int localMargin = std::min(width, height) / 10;
            Mat resultMask(height, width, CV_8UC1, Scalar::all(0));
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
                if(area > max_area){
                    printf("Area too large\n");
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
