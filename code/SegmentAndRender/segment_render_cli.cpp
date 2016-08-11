//
// Created by yanhang on 6/1/16.
//
#include "dynamicsegment.h"
#include <gflags/gflags.h>
#include "dynamicregularizer.h"
#include "../common/dynamicwarpping.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(resolution, 128, "resolution");
DEFINE_int32(tWindow, 100, "tWindow");
DEFINE_int32(downsample, 2, "downsample ratio");
DEFINE_string(classifierPath, "../../../data/traindata/visualword/model.rf", "Path to classifier");
DEFINE_string(codebookPath, "../../../data/traindata/visualword/metainfo_cluster00050.yml", "path to codebook");
DECLARE_string(flagfile);

void loadData(const FileIO& file_io, vector<Mat>& images, Mat& segMask, Depth& refDepth);

int main(int argc, char** argv) {
	char buffer[1024] = {};
	if(argc < 2){
		cerr << "Usage: ./SegmentAndRender <path-to-data>" << endl;
		return 1;
	}
	FileIO file_io(argv[1]);
	CHECK_GT(file_io.getTotalNum(), 0) << "Empty dataset";

	sprintf(buffer, "%s/config.txt", file_io.getDirectory().c_str());
	ifstream flagfile(buffer);
	if(flagfile.is_open()) {
		printf("Read flag from file\n");
		FLAGS_flagfile = string(buffer);
	}

	google::InitGoogleLogging(argv[1]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	printf("testFrame:%d, tWindow:%d\n", FLAGS_testFrame, FLAGS_tWindow);

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

    Mat seg_result_display, seg_result_flashy;
	segmentDisplay(file_io, FLAGS_testFrame, images, segMask, FLAGS_classifierPath, FLAGS_codebookPath ,seg_result_display);
	segmentFlashy(file_io, FLAGS_testFrame, images, seg_result_flashy);
//	return 0;

//	//visualize result
//    Mat seg_display, seg_flashy;
//    cv::resize(seg_result_display, seg_display, cv::Size(width, height), 0, 0, INTER_NEAREST);
//    cv::resize(seg_result_flashy, seg_flashy, cv::Size(width, height), 0, 0, INTER_NEAREST);
//    Mat seg_overlay(height, width, CV_8UC3, Scalar(0, 0, 0));
//    for (auto y = 0; y < height; ++y) {
//        for (auto x = 0; x < width; ++x) {
//            if (seg_display.at<int>(y, x) > 0)
//                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(255, 0, 0) * 0.6;
//            else if (seg_flashy.at<int>(y, x) > 0)
//                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(0, 255, 0) * 0.6;
//            else
//                seg_overlay.at<Vec3b>(y, x) = refImage.at<Vec3b>(y, x) * 0.4 + Vec3b(0, 0, 255) * 0.6;
//        }
//    }
//    sprintf(buffer, "%s/temp/segment%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame);
//    imwrite(buffer, seg_overlay);


	//////////////////////////////////////////////////////////
	//Rendering
	//reload full resolution image, set black pixel to (1,1,1)
	const double depthSmooth = 2.0;
	std::shared_ptr<DynamicWarpping> warping(new DynamicWarpping(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_resolution, depthSmooth));

	int offset = CHECK_NOTNULL(warping.get())->getOffset();
	images.resize((size_t)FLAGS_tWindow);
	for(auto v=0; v<FLAGS_tWindow; ++v){
		images[v] = imread(file_io.getImage(offset+v));
		for(auto y=0; y<images[v].rows; ++y){
			for(auto x=0; x<images[v].cols; ++x){
				if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
					images[v].at<Vec3b>(y,x) = Vec3b(1,1,1);
			}
		}
	}
	cv::resize(seg_result_display, seg_result_display, images[0].size(), 0, 0, INTER_NEAREST);
	cv::resize(seg_result_flashy, seg_result_flashy, images[0].size(), 0, 0, INTER_NEAREST);

	vector<vector<Vector2d> > segmentsDisplay;
	vector<vector<Vector2d> > segmentsFlashy;
	groupPixel(seg_result_display, segmentsDisplay);
	groupPixel(seg_result_flashy, segmentsFlashy);

	segmentsDisplay.insert(segmentsDisplay.end(), segmentsFlashy.begin(), segmentsFlashy.end());

    vector <Mat> finalResult;
    printf("Full warping...\n");
    warping->warpToAnchor(images, segmentsDisplay, segmentsFlashy, finalResult, FLAGS_tWindow);
    printf("Done\n");

    for (auto i = 0; i < finalResult.size(); ++i) {
        sprintf(buffer, "%s/temp/warped%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
                i);
        imwrite(buffer, finalResult[i]);
    }

//	printf("Running regularizaion\n");
//    vector <Mat> regulared;
//	float reg_t = (float)cv::getTickCount();
//	//dynamicRegularization(finalResult, segmentsDisplay, regulared, 0.6);
//	regularizationPoisson(finalResult, segmentsDisplay, regulared, 0.1, 0.5);
//	printf("Done, time usage: %.2fs\n", ((float)cv::getTickCount() -reg_t)/(float)cv::getTickFrequency());
//	CHECK_EQ(regulared.size(), finalResult.size());
//
//	vector<Mat> medianResult;
//	utility::temporalMedianFilter(finalResult, medianResult, 2);
//    utility::temporalMedianFilter(finalResult, regulared, 3);

//    for (auto i = 0; i < regulared.size(); ++i) {
//        sprintf(buffer, "%s/temp/regulared%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame,
//                i );
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
		sprintf(buffer, "%s/midres/prewarp/prewarpb%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, i);
		images[i] = imread(buffer);
		CHECK(images[i].data) << buffer;
	}
	CHECK(!images.empty());
//	width = images[0].cols;
//	height = images[0].rows;
	//segnet mask
//	sprintf(buffer, "%s/segnet/seg%05d.png", file_io.getDirectory().c_str(), FLAGS_testFrame);
//	Mat segMaskImg = imread(buffer);
//	CHECK(segMaskImg.data) << buffer;
//	cv::resize(segMaskImg, segMaskImg, cv::Size(width, height), 0,0, INTER_NEAREST);
//	//vector<Vec3b> validColor{Vec3b(0,0,128), Vec3b(128,192,192), Vec3b(128,128,192), Vec3b(128,128,128), Vec3b(0,128,128)};
//	vector<Vec3b> invalidColor{Vec3b(128,0,64), Vec3b(128,64,128), Vec3b(0,64,64), Vec3b(222,40,60)};
//
//	segMask.create(height, width, CV_8UC1);
//	for(auto y=0; y<height; ++y){
//		for(auto x=0; x<width; ++x){
//			Vec3b pix = segMaskImg.at<Vec3b>(y,x);
//			if(std::find(invalidColor.begin(), invalidColor.end(), pix) < invalidColor.end())
//				segMask.at<uchar>(y,x) = 0;
//			else
//				segMask.at<uchar>(y,x) = 255;
//		}
//	}
	//depth
	sprintf(buffer, "%s/midres/depth%05d.depth", file_io.getDirectory().c_str(), FLAGS_testFrame);
	CHECK(refDepth.readDepthFromFile(string(buffer))) << "Can not read depth file";

}