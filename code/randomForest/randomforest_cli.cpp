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
DEFINE_double(segLevel, 0.1, "segmentation level");
DEFINE_string(cache, "", "path to cached trianing data");
DEFINE_string(model, "", "path to model");
DEFINE_string(classifier, "rf", "rf or bt");
DEFINE_int32(treeDepth, -1, "max depth of the tree");
DEFINE_int32(numTree, -1, "number of trees");

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

	string arg;
	if(argc > 1)
		arg = string(argv[1]);

	if(FLAGS_mode == "train"){
		run_train(arg);
	}else if(FLAGS_mode == "test"){
		run_test(arg);
	}else if(FLAGS_mode == "detect"){
		run_detect(arg);
	}else{
		cerr << "Invalid mode" << endl;
		return 1;
	}
	return 0;
}

void run_train(const string& path) {
	string dir = path.substr(0, path.find_last_of("/"));
	dir.append("/");
	char buffer[128] = {};
	//prepare training data
	cv::Ptr<ml::TrainData> traindata = ml::TrainData::loadFromCSV(FLAGS_cache, 0);
	if (!traindata.get()) {
		CHECK(!path.empty());
		ifstream listIn(path.c_str());
		CHECK(listIn.is_open()) << "Can not open list file: " << path;
		string filename, gtname;

		Feature::FeatureOption option;
		Feature::TrainSet trainSet;

		while (listIn >> filename >> gtname) {
			vector<Mat> images;
			cv::VideoCapture cap(dir + filename);
			CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
			printf("Loading %s\n", filename.c_str());
			while (true) {
				Mat frame;
				if (!cap.read(frame))
					break;
				images.push_back(frame);
			}

			vector<Mat> segments;
			sprintf(buffer, "%s/segmentation/%s.pb", dir.c_str(), filename.c_str());
			segmentation::readSegmentAsMat(string(buffer), segments, FLAGS_segLevel);
			printf("Removing empty segments...\n");
			Feature::compressSegments(segments);

			Mat gtMask = imread(dir + gtname, false);
			CHECK(gtMask.data) << "Can not open ground truth: " << dir + gtname;

			Feature::extractFeature(images, segments, gtMask, option, trainSet);
		}
		printf("Number of positive: %d, number of negative: %d\n", (int) trainSet[1].size(), (int) trainSet[0].size());
		saveTrainData(FLAGS_cache, trainSet);
		traindata = convertTrainData(trainSet);
	}
	CHECK(traindata.get());
	printf("Training classifier, total samples:%d\n", traindata->getNSamples());
	cv::Ptr<ml::DTrees> forest;

	string classifier_path = FLAGS_model;
	if(classifier_path.empty()){
		classifier_path = dir + "model";
	}

	if(FLAGS_classifier == "rf"){
		classifier_path.append(".rf");
		forest = ml::RTrees::create();
		if(FLAGS_treeDepth > 0)
			forest->setMaxDepth(FLAGS_treeDepth);
		forest->train(traindata);
	}else if(FLAGS_classifier == "bt"){
		classifier_path.append(".bt");
		forest = ml::Boost::create();
		if(FLAGS_treeDepth > 0)
			forest->setMaxDepth(FLAGS_treeDepth);
		forest->train(traindata);
	}else{
		cerr << "Unsupported classifier." << endl;
		return;
	}
	printf("Saving %s\n", classifier_path.c_str());
	CHECK_NOTNULL(forest.get())->save(classifier_path);

}


void run_detect(const string& path){

}
void run_test(const string& path){
	cv::Ptr<ml::TrainData> testdata = cv::ml::TrainData::loadFromCSV(path, 0);
	CHECK(testdata.get()) << "Can not load test data: " << path;
	cv::Ptr<ml::DTrees> classifier;
	if(FLAGS_classifier == "rf"){
		classifier = ml::RTrees::load<ml::RTrees>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	}else if(FLAGS_classifier == "bt"){
		classifier = ml::Boost::load<ml::Boost>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	}
	Mat result;
	classifier->predict(testdata->getSamples(), result);
	const Mat& groundTruth = testdata->getResponses();
	CHECK_EQ(groundTruth.type(), CV_32F);
	CHECK_EQ(groundTruth.rows, result.rows);
	for(auto i=0; i<result.rows; ++i){

	}
}