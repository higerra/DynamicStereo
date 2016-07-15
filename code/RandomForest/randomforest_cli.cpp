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
DEFINE_double(segLevel, 0.2, "segmentation level");
DEFINE_string(cache, "", "path to cached trianing data");
DEFINE_string(model, "", "path to model");
DEFINE_string(classifier, "bt", "rf or bt");
DEFINE_int32(treeDepth, 15, "max depth of the tree");
DEFINE_int32(numTree, -1, "number of trees");
DEFINE_int32(compressLevel, -1, "-1: auto, 1: no compress, >1: compress");
DEFINE_int32(maxKFrame, 500, "if the video contains more than maxKFrame frames, compress");

void run_train(const string& path);
void run_detect(int argc, char** argv);
void run_test(const string& path);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage: ./RandomForest <path-to-data>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	CHECK_NE(FLAGS_compressLevel, 0);

	string arg;
	if(argc > 1)
		arg = string(argv[1]);

	if(FLAGS_mode == "train"){
		run_train(arg);
	}else if(FLAGS_mode == "test"){
		run_test(arg);
	}else if(FLAGS_mode == "detect"){
		run_detect(argc, argv);
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

	vector<float> levelList{0.1, 0.2,0.3};
	if (!traindata.get()) {
		CHECK(!path.empty());
		ifstream listIn(path.c_str());
		CHECK(listIn.is_open()) << "Can not open list file: " << path;
		string filename, gtname;

		Feature::FeatureOption option;
		Feature::TrainSet trainSet;

		while (listIn >> filename >> gtname) {
			int frameId = 0;
			int readInterval = 1;
			vector<Mat> images;
			cv::VideoCapture cap(dir + filename);
			double kFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);

			if(FLAGS_compressLevel < 0){
				if(kFrame > FLAGS_maxKFrame)
					readInterval = std::max((int)kFrame / FLAGS_compressLevel, 2);
			} else{
				readInterval = FLAGS_compressLevel;
			}
			CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
			printf("Loading %s, compress level: %d\n", filename.c_str(), readInterval);
			while (true) {
				Mat frame;
				if (!cap.read(frame))
					break;
				if (frameId % readInterval == 0) {
					images.push_back(frame);
				}
				frameId++;
			}

			printf("number of frames: %d/%d\n", (int)images.size(), (int)kFrame);
			vector<Mat> gradient(images.size());
			for(auto i=0; i<images.size(); ++i){
				Feature::computeGradient(images[i], gradient[i]);
				images[i].convertTo(images[i], CV_32FC3);
			}

			Mat gtMask = imread(dir + gtname, false);
			CHECK(gtMask.data) << "Can not open ground truth: " << dir + gtname;

			for(auto level: levelList) {
				printf("Segmentation level: %.3f\n", level);
				vector<Mat> segments_all, segments;
				sprintf(buffer, "%s/segmentation/%s.pb", dir.c_str(), filename.c_str());
				segmentation::readSegmentAsMat(string(buffer), segments_all, level);
				//don't forget to compress segments as well (if needed)
				for(auto i=0; i<segments_all.size(); ++i){
					if(i % readInterval == 0)
						segments.push_back(segments_all[i].clone());
				}
				segments_all.clear();

				Feature::compressSegments(segments);
				Feature::extractFeature(images, gradient, segments, gtMask, option, trainSet);
			}
		}
		printf("Number of positive: %d, number of negative: %d\n", (int) trainSet[1].size(), (int) trainSet[0].size());
		if(!FLAGS_cache.empty())
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
		if(FLAGS_model.empty())
			classifier_path.append(".rf");
		forest = ml::RTrees::create();
	}else if(FLAGS_classifier == "bt"){
		if(FLAGS_model.empty())
			classifier_path.append(".bt");
		forest = ml::Boost::create();
		if(FLAGS_numTree > 0)
			forest.dynamicCast<ml::Boost>()->setWeakCount(FLAGS_numTree);
		printf("Number of trees: %d\n", forest.dynamicCast<ml::Boost>()->getWeakCount());
	}else{
		cerr << "Unsupported classifier." << endl;
		return;
	}
	if(FLAGS_treeDepth > 0)
		forest->setMaxDepth(FLAGS_treeDepth);
	printf("Training. Max depth: %d\n", forest->getMaxDepth());
	forest->train(traindata);
	printf("Saving %s\n", classifier_path.c_str());
	CHECK_NOTNULL(forest.get())->save(classifier_path);

	double acc = testForest(traindata, forest);
	printf("Training accuracy: %.3f\n", acc);
}


void run_detect(int argc, char** argv) {
	if (argc < 3) {
		cerr << "Usage: ./RandomForest --mode=detect <path-to-video> <path-to-segmentation>" << endl;
		return;
	}
	printf("Loading data...\n");
	vector<Mat> images;
	cv::VideoCapture cap(argv[1]);
	CHECK(cap.isOpened()) << "Can not open input video: " << argv[1];

	while (true) {
		Mat frame;
		if (!cap.read(frame))
			break;
		images.push_back(frame);
	}

	Mat refImage = images[0].clone();

	vector<Mat> gradient(images.size());
	for(auto i=0; i<images.size(); ++i){
		Feature::computeGradient(images[i], gradient[i]);
		images[i].convertTo(images[i], CV_32FC3);
	}

	//empty ground truth
	Mat gt;
	Mat segmentVote(refImage.size(), CV_32FC1, Scalar::all(0.0f));

	printf("Running classification...\n");
	cv::Ptr<ml::DTrees> classifier;
	if (FLAGS_classifier == "rf") {
		classifier = ml::RTrees::load<ml::RTrees>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	} else if (FLAGS_classifier == "bt") {
		classifier = ml::Boost::load<ml::Boost>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	}
	printf("Max tree depth: %d\n", classifier->getMaxDepth());

	const vector<float> levelList{0.1,0.2,0.3};
	for(auto level: levelList) {
		printf("Level %.3f\n", level);
		vector<Mat> segments;
		segmentation::readSegmentAsMat(string(argv[2]), segments, level);
		int kSeg = Feature::compressSegments(segments);

		Feature::FeatureOption option;
		Feature::TrainSet testset;
		printf("Extracting feature...\n");
		Feature::extractFeature(images, gradient, segments, gt, option, testset);

		cv::Ptr<ml::TrainData> testPtr = convertTrainData(testset);
		CHECK(testPtr.get());
		Mat result;
		classifier->predict(testPtr->getSamples(), result);
		CHECK_EQ(result.rows, testset[0].size());
		vector<bool> segmentLabel((size_t) kSeg, false);
		for (auto i = 0; i < result.rows; ++i) {
			if (result.at<float>(i, 0) > 0.5) {
				int sid = testset[0][i].id;
				segmentLabel[sid] = true;
			}
		}
		for (auto y = 0; y < segments[0].rows; ++y) {
			for (auto x = 0; x < segments[0].cols; ++x) {
				for (auto v = 0; v < images.size(); ++v) {
					int sid = segments[v].at<int>(y, x);
					if (segmentLabel[sid]) {
						segmentVote.at<float>(y, x) += 1.0f;
					}
				}
			}
		}
	}//for(auto levelList)

	Mat mask(segmentVote.size(), CV_8UC3, Scalar(255, 0, 0));
	for (auto y = 0; y < mask.rows; ++y) {
		for (auto x = 0; x < mask.cols; ++x) {
			if (segmentVote.at<float>(y, x) > (float)levelList.size() * (float) images.size() / 2)
				mask.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		}
	}
	const double blend_weight = 0.4;
	Mat vis;
	cv::addWeighted(refImage, blend_weight, mask, 1.0 - blend_weight, 0.0, vis);

	string fullPath = string(argv[1]);
	string dir = fullPath.substr(0, fullPath.find_last_of('/')+1);
	string filename = fullPath.substr(fullPath.find_last_of('/')+1, fullPath.find_last_of('.'));

	imwrite(dir+filename+".png", vis);

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
	printf("Max tree depth: %d\n", classifier->getMaxDepth());

	double acc = testForest(testdata, classifier);
	printf("Test accuracy: %.3f\n", acc);
}
