//
// Created by yanhang on 7/27/16.
//

#include "videosegmentation.h"
#include "../external/segment_gb/segment-image.h"

using namespace std;
using namespace cv;
using namespace segment_gb;

namespace dynamic_stereo {

    void edgeAggregation(const VideoMat &input, cv::Mat &output){
        CHECK(!input.empty());
        output.create(input[0].size(), CV_32FC1);
        output.setTo(cv::Scalar::all(0));
        for (auto i =0; i<input.size(); ++i) {
            cv::Mat edge_sobel(input[i].size(), CV_32FC1, cv::Scalar::all(0));
            cv::Mat gray, gx, gy;
            cvtColor(input[i], gray, CV_BGR2GRAY);
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

    int segment_video(const std::vector<cv::Mat>& input, cv::Mat& output,
                      const int smoothSize, const float c, const float theta, const int min_size){
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;

        printf("preprocessing\n");
        std::vector<cv::Mat> smoothed(input.size());
        for(auto v=0; v<input.size(); ++v){
            cv::Mat temp;
            input[v].convertTo(temp, cv::DataType<float>::type);
            cv::blur(temp, smoothed[v], cv::Size(smoothSize, smoothSize));
        }

        cv::Mat edgeMap;
        edgeAggregation(smoothed, edgeMap);

        const int stride1 = 8;
        const int stride2 = (int)input.size() / 2;


        std::shared_ptr<PixelFeatureExtractorBase> pixel_extractor(new PixelValue());
        std::shared_ptr<DistanceMetricBase<float> > pixel_comparator(new DistanceL2<float>());
        std::shared_ptr<TemporalFeatureExtractorBase> temporal_extractor(
                new TransitionPattern(pixel_extractor.get(), pixel_comparator.get(), stride1, stride2, theta));

        std::shared_ptr<DistanceMetricBase<float> > feature_comparator(new AverageL1<float>());

        printf("Computing edge weight\n");
        vector<vector<float> > features;
        temporal_extractor->extractVideo(smoothed, features);
        // build graph
        std::vector<edge> edges((size_t)width*height*4);
        int num = 0;
        //8 neighbor
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float edgeness = edgeMap.at<float>(y,x);
                if (x < width - 1) {
                    edges[num].a = y * width + x;
                    edges[num].b = y * width + (x + 1);
                    edges[num].w = feature_comparator->evaluate(features[edges[num].a], features[edges[num].b]) * edgeness;
                    num++;
                }

                if (y < height - 1) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y + 1) * width + x;
                    edges[num].w = feature_comparator->evaluate(features[edges[num].a], features[edges[num].b]) * edgeness;
                    num++;
                }

                if ((x < width - 1) && (y < height - 1)) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y + 1) * width + (x + 1);
                    edges[num].w = feature_comparator->evaluate(features[edges[num].a], features[edges[num].b]) * edgeness;
                    num++;
                }

                if ((x < width - 1) && (y > 0)) {
                    edges[num].a = y * width + x;
                    edges[num].b = (y - 1) * width + (x + 1);
                    edges[num].w = feature_comparator->evaluate(features[edges[num].a], features[edges[num].b]) * edgeness;
                    num++;
                }
            }
        }

        printf("segment graph\n");

        std::unique_ptr<universe> u(segment_graph(width * height, edges, c));

        printf("post processing\n");
        // post process small components
        for (int i = 0; i < num; i++) {
            int a = u->find(edges[i].a);
            int b = u->find(edges[i].b);
            if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
                u->join(a, b);
        }

        output = cv::Mat(height, width, CV_32S, cv::Scalar::all(0));

        //remap labels
        std::vector<std::pair<int, int> > labelMap((size_t)width * height);
        int curMaxLabel = -1;
        int nLabel = -1;
        for (int i=0; i<width * height; ++i) {
            int comp = u->find(i);
            CHECK_LT(comp, width * height);
            labelMap[i] = std::pair<int,int>(comp, i);
        }
        std::sort(labelMap.begin(), labelMap.end());
        for(auto i=0; i<labelMap.size(); ++i){
            CHECK_GE(labelMap[i].first, 0);
            if(labelMap[i].first > curMaxLabel){
                curMaxLabel = labelMap[i].first;
                nLabel++;
            }
            int pixId = labelMap[i].second;
            output.at<int>(pixId/width, pixId%width) = nLabel;
        }

        nLabel++;
        return nLabel;
    }

    cv::Mat visualizeSegmentation(const cv::Mat& input){
        return segment_gb::visualizeSegmentation(input);
    }

}//namespace dynamic_stereo