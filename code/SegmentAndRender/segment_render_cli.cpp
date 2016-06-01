//
// Created by yanhang on 6/1/16.
//
#include "dynamicsegment.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

int main(int argc, char** argc) {
    printf("Segmenting...\n");
    vector <vector<Vector2d>> segmentsDisplay;
    vector <vector<Vector2d>> segmentsFlashy;
    shared_ptr <DynamicSegment> segment(
            new DynamicSegment(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_downsample, depths, depthInd));

    Mat seg_result_display;
    Mat seg_result_flashy;
//segment->segmentFlashy(prewarp1, seg_result_flashy);
    segment->segmentDisplay(prewarp1, segMask, FLAGS_classifierPath, seg_result_display, segmentsDisplay);

    segment.reset();
    Mat seg_display, seg_flashy;
    cv::resize(seg_result_display, seg_display, cv::Size(width, height), 0, 0, INTER_NEAREST);
    cv::resize(seg_result_flashy, seg_flashy, cv::Size(width, height), 0, 0, INTER_NEAREST);
    Mat seg_overlay(height, width, CV_8UC3, Scalar(0, 0, 0));
    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            if (seg_display.at<int>(y, x) > 0)
                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(255, 0, 0) * 0.6;
            else if (seg_flashy.at<int>(y, x) > 0)
                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(0, 255, 0) * 0.6;
            else
                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(0, 0, 255) * 0.6;
        }
    }
    sprintf(buffer, "%s/temp/segment%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame);
    imwrite(buffer, seg_overlay);

    vector <Mat> finalResult;
    printf("Full warping...\n");
    warpping->warpToAnchor(segmentsDisplay, segmentsFlashy, finalResult, FLAGS_tWindow);
    printf("Done\n");
    for (auto i = 0; i < finalResult.size(); ++i) {
        sprintf(buffer, "%s/temp/warped%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
                i + warpping->getOffset());
        imwrite(buffer, finalResult[i]);
    }

//	printf("Running regularizaion\n");
    vector <Mat> regulared;
//	float reg_t = (float)cv::getTickCount();
//	dynamicRegularization(finalResult, segmentsDisplay, regulared, 0.6);
//	printf("Done, time usage: %.2fs\n", ((float)cv::getTickCount() -reg_t)/(float)cv::getTickFrequency());
//	CHECK_EQ(regulared.size(), finalResult.size());
//	vector<Mat> medianResult;
//	utility::temporalMedianFilter(finalResult, medianResult, 2);
    utility::temporalMedianFilter(finalResult, regulared, 3);

    for (auto i = 0; i < regulared.size(); ++i) {
        sprintf(buffer, "%s/temp/regulared%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
                i + warpping->getOffset());
        imwrite(buffer, regulared[i]);
    }
    return 0;
}