//
// Created by yanhang on 5/28/16.
//

#include "gtest/gtest.h"
#include "CVdescriptor.h"
#include "regiondescriptor.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

TEST(RegionFeature, shape){
    Mat img = imread("test.png", false);
    CHECK(img.data);
    const int width = img.cols;
    const int height = img.rows;

    Mat components, centroid, stat;
    cv::connectedComponentsWithStats(img, components, stat, centroid);
    vector<Feature::PixelGroup> pixelGroups;
    Feature::regroupSegments(components, pixelGroups);

    for(auto i=1; i<pixelGroups.size(); ++i){
        printf("Segment %d, centroid: (%.2f,%.2f)\t", i,
               centroid.at<double>(i,0), centroid.at<double>(i,1));
        vector<float> desc;
        Feature::computeShape(pixelGroups[i], width, height, desc);
        printf("descriptor: ");
        for(auto v: desc)
            printf("%.3f ", v);
        printf("\n");
    }
}