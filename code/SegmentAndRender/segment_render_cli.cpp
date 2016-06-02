//
// Created by yanhang on 6/1/16.
//
#include "dynamicsegment.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(downsample, 4, "downsample");
DEFINE_int32(tWindow, 80, "tWindow");
DEFINE_string(classifierPath, "../../../data/svmTrain/model_newyorkRGB.svm", "Path to classifier");

void loadData(const FileIO& file_io, vector<Mat>& images, Mat& segMask, Depth& refDepth);

int main(int argc, char** argv) {
	char buffer[1024] = {};
	google::InitGoogleLogging(argv[1]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	if(argc < 2){
		cerr << "Usage: ./SegmentAndRender <path-to-data>" << endl;
		return 1;
	}
	FileIO file_io(argv[1]);
	CHECK_GT(file_io.getTotalNum(), 0) << "Empty dataset";

	vector<Mat> images;
	Mat segMask;
	Depth refDepth;
	int width, height;
	printf("Loading...\n");
	loadData(file_io, images, segMask, refDepth);
	width = images[0].cols;
	height = images[0].rows;
	Mat refImage = imread(file_io.getImage(FLAGS_testFrame));
	cv::resize(refImage, refImage, cv::Size(width, height), cv::INTER_CUBIC);

	////////////////////////////////////////////
	//Segmentation
    printf("Segmenting...\n");
    vector <vector<Vector2d>> segmentsDisplay;
    vector <vector<Vector2d>> segmentsFlashy;

    Mat seg_result_display;
    Mat seg_result_flashy;
	segmentFlashy(file_io, FLAGS_testFrame, images, seg_result_flashy);
    segmentDisplay(file_io, FLAGS_testFrame, images, segMask, FLAGS_classifierPath, seg_result_display, segmentsDisplay);

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


	//////////////////////////////////////////////////////////
	//Rendering
//    vector <Mat> finalResult;
//    printf("Full warping...\n");
//    warpping->warpToAnchor(segmentsDisplay, segmentsFlashy, finalResult, FLAGS_tWindow);
//    printf("Done\n");
//    for (auto i = 0; i < finalResult.size(); ++i) {
//        sprintf(buffer, "%s/temp/warped%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
//                i + warpping->getOffset());
//        imwrite(buffer, finalResult[i]);
//    }

//	printf("Running regularizaion\n");
//    vector <Mat> regulared;
//	float reg_t = (float)cv::getTickCount();
//	dynamicRegularization(finalResult, segmentsDisplay, regulared, 0.6);
//	printf("Done, time usage: %.2fs\n", ((float)cv::getTickCount() -reg_t)/(float)cv::getTickFrequency());
//	CHECK_EQ(regulared.size(), finalResult.size());
//	vector<Mat> medianResult;
//	utility::temporalMedianFilter(finalResult, medianResult, 2);
//    utility::temporalMedianFilter(finalResult, regulared, 3);
//
//    for (auto i = 0; i < regulared.size(); ++i) {
//        sprintf(buffer, "%s/temp/regulared%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
//                i + warpping->getOffset());
//        imwrite(buffer, regulared[i]);
//    }
    return 0;
}

void loadData(const FileIO& file_io, vector<Mat>& images, Mat& segMask, Depth& refDepth){
	//images
	char buffer[1024] = {};
	images.resize((size_t)FLAGS_tWindow);
	int width, height;
	const int offset = FLAGS_testFrame - FLAGS_tWindow / 2;
	CHECK_GE(offset, 0);
	for(auto i=0; i<FLAGS_tWindow; ++i){
		sprintf(buffer, "%s/midres/prewarp/prewarpb%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, i+offset);
		images[i] = imread(buffer);
		CHECK(images[i].data) << buffer;
	}
	CHECK(!images.empty());
	width = images[0].cols;
	height = images[0].rows;
	//segnet mask
	sprintf(buffer, "%s/segnet/seg%05d.png", file_io.getDirectory().c_str(), FLAGS_testFrame);
	Mat segMaskImg = imread(buffer);
	CHECK(segMaskImg.data) << buffer;
	cv::resize(segMaskImg, segMaskImg, cv::Size(width, height), 0,0, INTER_NEAREST);
	//vector<Vec3b> validColor{Vec3b(0,0,128), Vec3b(128,192,192), Vec3b(128,128,192), Vec3b(128,128,128), Vec3b(0,128,128)};
	vector<Vec3b> invalidColor{Vec3b(128,0,64), Vec3b(128,64,128), Vec3b(0,64,64), Vec3b(222,40,60)};

	segMask.create(height, width, CV_8UC1);
	for(auto y=0; y<height; ++y){
		for(auto x=0; x<width; ++x){
			Vec3b pix = segMaskImg.at<Vec3b>(y,x);
			if(std::find(invalidColor.begin(), invalidColor.end(), pix) < invalidColor.end())
				segMask.at<uchar>(y,x) = 0;
		}
	}
	//depth
	sprintf(buffer, "%s/midres/depth%05d.depth", file_io.getDirectory().c_str(), FLAGS_testFrame);
	CHECK(refDepth.readDepthFromFile(string(buffer))) << "Can not read depth file";

}