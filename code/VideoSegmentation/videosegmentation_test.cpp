//
// Created by yanhang on 7/31/16.
//

#include <opencv2/xfeatures2d.hpp>
#include "gtest/gtest.h"
#include "videosegmentation.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

TEST(Feature, BRIEFWrapper){
    shared_ptr<BRIEFWrapper> brief(new BRIEFWrapper());
    Mat image = imread("brief_test.png");
    CHECK(image.data);

    float start_t = (float)cv::getTickCount();
    Mat feats;
    brief->extractAll(image, feats);
    printf("Time: %.3fs\n", ((float)getTickCount() - start_t) / (float)cv::getTickFrequency());
    EXPECT_EQ(feats.rows, image.rows * image.cols);
}

