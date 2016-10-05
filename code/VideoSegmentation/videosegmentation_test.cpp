//
// Created by yanhang on 7/31/16.
//

#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Eigen>

#include "gtest/gtest.h"
#include "distance_metric.h"

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