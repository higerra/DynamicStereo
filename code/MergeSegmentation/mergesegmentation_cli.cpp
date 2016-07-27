//
// Created by yanhang on 7/27/16.
//

#include "mergesegmentation.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./MergeSegmentation <path-to-video>" << endl;
        return 1;
    }

    //load video
    printf("Loading video...\n");
    VideoCapture cap(argv[1]);
    CHECK(cap.isOpened()) << "Can not load video: " << argv[1];
    vector<Mat> images;
    while(true){
        Mat frame;
        if(!cap.read(frame))
            break;
        images.push_back(frame);
    }

    vector<int> test_smoothSize{3,5,7,9};
    cv::namedWindow("segment_gb");
    for(auto ss: test_smoothSize) {
        Mat segment_test;
        vector<vector<int> > segGroup;
        segment_gb::segment_image(images[0], segment_test, segGroup, ss, 300, 100);
        Mat segment_vis = segment_gb::visualizeSegmentation(segment_test);
        imshow("segment_gb", segment_vis);
        waitKey(0);
    }

    //edge aggregation
//    printf("Computing edges...\n");
//    Mat edge;
//    edgeAggregation(images, edge);

    return 0;
}

