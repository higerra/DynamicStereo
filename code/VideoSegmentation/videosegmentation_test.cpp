//
// Created by yanhang on 7/31/16.
//

#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Eigen>

#include "gtest/gtest.h"
#include "distance_metric.h"
#include "region_feature.h"
#include "videosegmentation.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

//TEST(Eigen, replicate){
//    Vector3d m = Vector3d::Random();
//    Matrix<double, 10, 3> me = m.transpose().replicate(10,1);
//    cout << m << endl;
//    cout << me << endl;
//}

class VideoTest: public ::testing::Test{
public:
    VideoTest(): s1(8), s2(4), theta(100){}
protected:
    virtual void SetUp(){
        LOG(INFO) << "Setting up test";
        string path = "warped00100.avi";
        cv::VideoCapture cap(path);
        CHECK(cap.isOpened()) << "Can not open video " << path;
        while(true){
            Mat frame;
            if(!cap.read(frame))
                break;
            Mat small;
            cv::pyrDown(frame, small);
            images.push_back(small);
        }
    }
    vector<Mat> images;

    const int s1;
    const int s2;
    const float theta;
};

TEST(Distance, CombinedWeighting) {
    const int N = 10;
    vector<std::shared_ptr<DistanceMetricBase> > comparators{
            std::shared_ptr<DistanceMetricBase>(new DistanceL1()),
            std::shared_ptr<DistanceMetricBase>(new DistanceL2())
    };
    vector<double> weights{0.2, 0.8};
    vector<size_t> splits{5};

    std::shared_ptr<DistanceMetricBase> combined_distance(
            new DistanceCombinedWeighting(splits, weights, comparators));

    Mat a1(N, 1, CV_64FC1), a2(N, 1, CV_64FC1);
    cv::RNG rng((unsigned long)time(NULL));
    rng.fill(a1, cv::RNG::UNIFORM, Scalar(0), Scalar(10));
    rng.fill(a2, cv::RNG::UNIFORM, Scalar(0), Scalar(10));

    double distance_eval = combined_distance->evaluate(a1, a2);

    double distance_gt = 0.0;
    vector<double> distance_parts(weights.size(), 0.0);
    for(auto i=0; i<splits[0]; ++i){
        distance_parts[0] += std::fabs(a1.at<double>(i, 0) - a2.at<double>(i,0));
    }
    double dis_part1_eval = comparators[0]->evaluate(a1.rowRange(0, splits[0]), a2.rowRange(0, splits[0]));
    EXPECT_NEAR(dis_part1_eval, distance_parts[0], numeric_limits<double>::epsilon());

    for(auto i=splits[0]; i<N; ++i){
        distance_parts[1] += (a1.at<double>(i,0) - a2.at<double>(i,0)) * (a1.at<double>(i,0) - a2.at<double>(i,0));
    }
    distance_parts[1] = std::sqrt(distance_parts[1]);
    double dis_part2_eval = comparators[1]->evaluate(a1.rowRange(splits[0], N), a2.rowRange(splits[0], N));
    EXPECT_NEAR(dis_part2_eval, distance_parts[1], numeric_limits<double>::epsilon());

    distance_gt = distance_parts[0] * weights[0] + distance_parts[1] * weights[1];

    EXPECT_NEAR(distance_eval, distance_gt, std::numeric_limits<double>::epsilon());
}

//TEST_F(VideoTest, RegionTransition){
//    video_segment::PixelValue pixel_extractor;
//    vector<Mat> pixel_features(images.size());
//    LOG(INFO) << "Extracting pixel features";
//
//    for(auto v=0; v<images.size(); ++v){
//        pixel_extractor.extractAll(images[v], pixel_features[v]);
//    }
//
//    LOG(INFO) << "Computing pixel transition";
//    Mat output_pixel_transition;
//    video_segment::TransitionPattern transition_pattern(images.size(), s1, s2, theta, pixel_extractor.getDefaultComparator());
//    transition_pattern.computeFromPixelFeature(pixel_features, output_pixel_transition);
//    LOG(INFO) << "Norm of pixel_transition: " << cv::norm(output_pixel_transition, cv::NORM_L1);
//
//    vector<video_segment::Region> fake_regions(pixel_features[0].rows, video_segment::Region());
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        fake_regions[rid].pix_id.push_back(rid);
//    }
//
//    vector<video_segment::Region*> regions_ptr(fake_regions.size(), nullptr);
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        regions_ptr[rid] = &(fake_regions[rid]);
//    }
//
//    video_segment::TemporalAverage average;
//    video_segment::RegionTransitionPattern region_transition_pattern(
//            images.size(), s1, s2, theta, average.getDefaultComparator(), &average);
//
//    LOG(INFO) << "Computing region transition";
//    Mat output_region_transition;
//    region_transition_pattern.ExtractFromPixelFeatures(pixel_features, regions_ptr, output_region_transition);
//
//    EXPECT_NEAR(cv::norm(output_pixel_transition, output_region_transition, cv::NORM_L1), 0.0, numeric_limits<double>::epsilon());
//}

//TEST_F(VideoTest, RegionHistogram){
//    video_segment::PixelValue pixel_extractor;
//    vector<Mat> pixel_features(images.size());
//    LOG(INFO) << "Extracting pixel features";
//
//    for(auto v=0; v<images.size(); ++v){
//        pixel_extractor.extractAll(images[v], pixel_features[v]);
//    }
//
//    vector<int> kBins{8,8,8};
//    const int width = images[0].cols;
//    const int height = images[0].rows;
//    video_segment::ColorHistogram::ColorSpace cspace = video_segment::ColorHistogram::LAB;
//
//    LOG(INFO) << "Computing pixel transition";
//    Mat output_pixel_hist;
//    video_segment::ColorHistogram color_histogram(cspace, kBins, width, height, 0);
//    color_histogram.computeFromPixelFeature(pixel_features, output_pixel_hist);
//    LOG(INFO) << "Norm of pixel_hist: " << cv::norm(output_pixel_hist, cv::NORM_L1);
//
//    vector<video_segment::Region> fake_regions(pixel_features[0].rows, video_segment::Region());
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        fake_regions[rid].pix_id.push_back(rid);
//    }
//
//    vector<video_segment::Region*> regions_ptr(fake_regions.size(), nullptr);
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        regions_ptr[rid] = &(fake_regions[rid]);
//    }
//
//    video_segment::RegionColorHist region_color_hist(cspace, kBins, width, height);
//
//    LOG(INFO) << "Computing region transition";
//    Mat output_region_hist;
//    region_color_hist.ExtractFromPixelFeatures(pixel_features, regions_ptr, output_region_hist);
//    LOG(INFO) << "Norm of region_hist: " << cv::norm(output_region_hist, cv::NORM_L1);
//
//    EXPECT_NEAR(cv::norm(output_pixel_hist, output_region_hist, cv::NORM_L1), 0.0, numeric_limits<double>::epsilon());
//}

//TEST_F(VideoTest, RegionCombined){
//    video_segment::PixelValue pixel_extractor;
//    vector<Mat> pixel_features(images.size());
//    LOG(INFO) << "Extracting pixel features";
//
//    for(auto v=0; v<images.size(); ++v){
//        pixel_extractor.extractAll(images[v], pixel_features[v]);
//    }
//
//    vector<int> kBins{8,8,8};
//    const int width = images[0].cols;
//    const int height = images[0].rows;
//    video_segment::ColorHistogram::ColorSpace cspace = video_segment::ColorHistogram::LAB;
//    const int stride1 = 8;
//    const int stride2 = 4;
//    const float theta = 100;
//
//    LOG(INFO) << "Computing pixel combined feature";
//    Mat output_pixel_combined;
//    vector<shared_ptr<video_segment::TemporalFeatureExtractorBase> > temporal_extractors(2);
//    temporal_extractors[0].reset(new video_segment::ColorHistogram(cspace, kBins, width, height, 0));
//    temporal_extractors[1].reset(new video_segment::TransitionPattern((int)images.size(), stride1, stride2, theta,
//                                                                      pixel_extractor.getDefaultComparator()));
//    video_segment::CombinedTemporalFeature combined_pixel(temporal_extractors, {0.5,0.5});
//    combined_pixel.computeFromPixelFeatures({pixel_features, pixel_features}, output_pixel_combined);
//    LOG(INFO) << "Norm of pixel_combined: " << cv::norm(output_pixel_combined, cv::NORM_L1);
//
//    vector<video_segment::Region> fake_regions(pixel_features[0].rows, video_segment::Region());
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        fake_regions[rid].pix_id.push_back(rid);
//    }
//
//    vector<video_segment::Region*> regions_ptr(fake_regions.size(), nullptr);
//    for(auto rid=0; rid < fake_regions.size(); ++rid){
//        regions_ptr[rid] = &(fake_regions[rid]);
//    }
//
//    vector<shared_ptr<video_segment::RegionFeatureExtractorBase> > region_extractors(2);
//    video_segment::TemporalAverage average_extractor;
//    region_extractors[0].reset(new video_segment::RegionColorHist(cspace, kBins, width, height));
//    region_extractors[1].reset(new video_segment::RegionTransitionPattern((int)images.size(), stride1, stride2, theta,
//                                                                          pixel_extractor.getDefaultComparator(),
//                                                                          &average_extractor));
//    video_segment::RegionCombinedFeature combined_region(region_extractors, {0.5,0.5});
//    LOG(INFO) << "Computing region combined";
//    Mat output_region_combined;
//    combined_region.ExtractFromPixelFeatureArray({pixel_features, pixel_features}, regions_ptr, output_region_combined);
//    LOG(INFO) << "Norm of region_combined: " << cv::norm(output_region_combined, cv::NORM_L1);
//
//    EXPECT_NEAR(cv::norm(output_pixel_combined, output_region_combined, cv::NORM_L1), 0.0, numeric_limits<double>::epsilon());
//}

TEST(VideoSegment, BuildEdgeMap){
    const int width = 60;
    const int height = 60;
    vector<video_segment::Region> toy_regions(3);
    for(auto rid=0; rid < 3; ++rid) {
        for (auto y = rid * height / 3; y < (rid + 1) * height / 3; ++y) {
            for (auto x = 0; x < width; ++x) {
                toy_regions[rid].pix_id.push_back(y * width + x);
            }
        }
    }

    vector<video_segment::Region*> region_ptr(toy_regions.size());
    for(auto rid=0; rid < toy_regions.size(); ++rid){
        region_ptr[rid] = &toy_regions[rid];
    }

    vector<segment_gb::edge> edge_map;
    video_segment::BuildEdgeMap(region_ptr, edge_map, width, height);
    ASSERT_TRUE(!edge_map.empty());
    for(const auto& e: edge_map){
        printf("%d,%d,%.3f\n", e.a, e.b, e.w);
    }
}