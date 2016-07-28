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
	google::InitGoogleLogging(argv[0]);

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

    //test for video segmentation based on binary descriptor
	Mat segment;
	printf("Video segmentation...\n");
	segment_gb::segment_video(images, segment, 9, 1.5, 100, 100);
	Mat segment_vis = segment_gb::visualizeSegmentation(segment);
	Mat result;
	cv::addWeighted(images[0], 0.1, segment_vis, 0.9, 0.0, result);
	imshow("video segmentation", result);
	waitKey(0);

    //edge aggregation
//    printf("Computing edges...\n");
//    Mat edge;
//    edgeAggregation(images, edge);

    return 0;
}

