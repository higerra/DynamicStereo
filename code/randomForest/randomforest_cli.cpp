//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"
#include "../external/video_segmentation/segment_util/segmentation_io.h"
#include "../external/video_segmentation/segment_util/segmentation_util.h"
#include <fstream>
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_string(mode, "train", "mode");
DEFINE_double(segLevel, 0.3, "segmentation level");

void run_train(const string& path);
void run_detect(const string& path);
void run_test(const string& path);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage: ./RandomForest <path-to-data>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	if(argc < 2){
		cerr << "Usage: ./RandomForest <path-to-data>" << endl;
		return 1;
	}

	if(FLAGS_mode == "train"){
		run_train(string(argv[1]));
	}else if(FLAGS_mode == "test"){
		run_test(string(argv[1]));
	}else if(FLAGS_mode == "detect"){
		run_detect(string(argv[1]));
	}else{
		cerr << "Invalid mode" << endl;
		return 1;
	}
	return 0;
}

void run_train(const string& path){
	string dir = path.substr(0, path.find_last_of("/"));
	dir.append("/");

	char buffer[128] = {};
	ifstream listIn(path.c_str());
	CHECK(listIn.is_open()) << "Can not open list file: " << path;
	string filename, gtname;

	FeatureOption option;
	TrainSet trainSet;

	while(listIn >> filename >> gtname){
		vector<Mat> images;
		cv::VideoCapture cap(dir+filename);
		CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
		printf("Loading %s\n", filename.c_str());
		while(true){
			Mat frame;
			if(!cap.read(frame))
				break;
			images.push_back(frame);
		}

		vector<Mat> segments;
		sprintf(buffer, "%s/segmentation/%s.pb", dir.c_str(), filename.c_str());
		segmentation::readSegmentAsMat(string(buffer), segments, FLAGS_segLevel);

		Mat gtMask = imread(dir+gtname);
		CHECK(gtMask.data) << "Can not open ground truth: " << dir + gtname;


		extractFeature(images, segments, gtMask, option, trainSet);
	}

}
void run_detect(const string& path){

}
void run_test(const string& path){

}