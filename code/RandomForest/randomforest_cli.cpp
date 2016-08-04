//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"
#include "../VideoSegmentation/videosegmentation.h"
#include <fstream>
#include <gflags/gflags.h>
#include "../MLModule/detectimage.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_string(mode, "train", "mode");
DEFINE_double(segLevel, 0.2, "segmentation level");
DEFINE_string(cache, "", "path to cached trianing data");
DEFINE_string(model, "", "path to model");
DEFINE_string(validation, "", "path to validation dataset");

DEFINE_string(classifier, "bt", "rf or bt");
DEFINE_int32(treeDepth, 15, "max depth of the tree");
DEFINE_int32(numTree, -1, "number of trees");

cv::Ptr<ml::DTrees> run_train(cv::Ptr<ml::TrainData> traindata);
void run_detect(int argc, char** argv);
void run_test(const string& path);
cv::Ptr<ml::TrainData> run_extract(const std::string& path);

static const int smoothSize = 9;
static const int minSize = 300;
static const float theta = 100;
static const vector<float> levelList{10, 20, 30};

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage: ./RandomForest <path-to-data>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	string arg, dir;
	if(argc > 1) {
        arg = string(argv[1]);
        dir = arg.substr(0, arg.find_last_of("."));
        dir.append("/");
    }
	if(FLAGS_mode == "train"){
		cv::Ptr<ml::TrainData> traindata = run_extract(arg);
        //compose output name
        string classifier_path = FLAGS_model;
        if(classifier_path.empty()){
            classifier_path = dir + "model";
        }
        if(FLAGS_model.empty()) {
            if (FLAGS_classifier == "rf") {
                classifier_path.append(".rf");
            } else if (FLAGS_classifier == "bt") {
                classifier_path.append(".bt");
            } else {
                cerr << "Unsupported classifier." << endl;
                return 1;
            }
        }
        printf("Trainig...\n");
        cv::Ptr<ml::DTrees> classifier = run_train(traindata);
        printf("Done...\n");
        CHECK_NOTNULL(classifier.get())->save(classifier_path);
        double acc = testForest(traindata, classifier);
        printf("Training accuracy: %.3f\n", acc);

		cv::Ptr<ml::TrainData> valdata = cv::ml::TrainData::loadFromCSV(FLAGS_validation, 0);
		if(valdata.get()){
			double val_acc = testForest(valdata, classifier);
			printf("Validation accuracy: %.3f\n", val_acc);
		}
	}else if(FLAGS_mode=="extract"){
        cv::Ptr<ml::TrainData> traindata = run_extract(arg);
    } else if(FLAGS_mode == "test"){
		run_test(arg);
	}else if(FLAGS_mode == "detect"){
		run_detect(argc, argv);
	}else{
		cerr << "Invalid mode" << endl;
		return 1;
	}
	return 0;
}

cv::Ptr<ml::TrainData> run_extract(const std::string& path){
	//prepare training data
	cv::Ptr<ml::TrainData> traindata = ml::TrainData::loadFromCSV(FLAGS_cache, 0);
	if (!traindata.get()) {
		string dir = path.substr(0, path.find_last_of("/"));
		dir.append("/");
		char buffer[128] = {};

		CHECK(!path.empty());
		ifstream listIn(path.c_str());
		CHECK(listIn.is_open()) << "Can not open list file: " << path;
		string filename, gtname;
		ML::TrainSet trainSet;

		while (listIn >> filename >> gtname) {
			vector<Mat> images;
			cv::VideoCapture cap(dir + filename);
			CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
			printf("Loading %s\n", (dir+filename).c_str());
			while (true) {
				Mat frame;
				if (!cap.read(frame))
					break;
				images.push_back(frame);
			}
			vector<Mat> gradient(images.size());
			for(auto i=0; i<images.size(); ++i){
                ML::MLUtility::computeGradient(images[i], gradient[i]);
				images[i].convertTo(images[i], CV_32FC3);
			}

			Mat gtMask = imread(dir + gtname, false);
			CHECK(gtMask.data) << "Can not open ground truth: " << dir + gtname;
			cv::resize(gtMask, gtMask, images[0].size(), 0, 0, INTER_NEAREST);

			for(auto level: levelList) {
				printf("Segmentation level: %.3f\n", level);
                Mat segments;
                printf("Segmenting...\n");
                video_segment::segment_video(images, segments, level);
				CHECK(segments.data);
                printf("Extracting feature...\n");
				ML::extractFeature(images, gradient, segments, gtMask, trainSet);
			}
		}
		printf("Number of positive: %d, number of negative: %d\n", (int) trainSet[1].size(), (int) trainSet[0].size());
		traindata = ML::MLUtility::convertTrainData(trainSet);
        if(!FLAGS_cache.empty())
            ML::MLUtility::writeTrainData(FLAGS_cache, traindata);
	}
	return traindata;
}

cv::Ptr<ml::DTrees> run_train(cv::Ptr<ml::TrainData> traindata) {
	CHECK(traindata.get());
	printf("Training classifier, total samples:%d\n", traindata->getNSamples());
	cv::Ptr<ml::DTrees> forest;
	if(FLAGS_classifier == "rf"){
		forest = ml::RTrees::create();
        if(FLAGS_numTree > 0) {
            forest.dynamicCast<ml::RTrees>()->
                    setTermCriteria(
                    cv::TermCriteria(TermCriteria::MAX_ITER, FLAGS_numTree, std::numeric_limits<double>::min()));
        }
	}else if(FLAGS_classifier == "bt"){
		forest = ml::Boost::create();
		if(FLAGS_numTree > 0)
			forest.dynamicCast<ml::Boost>()->setWeakCount(FLAGS_numTree);
		printf("Number of trees: %d\n", forest.dynamicCast<ml::Boost>()->getWeakCount());
	}else{
		cerr << "Unsupported classifier." << endl;
		return cv::Ptr<ml::DTrees>();
	}
	if(FLAGS_treeDepth > 0)
		forest->setMaxDepth(FLAGS_treeDepth);
	printf("Max depth: %d\n", forest->getMaxDepth());
	forest->train(traindata);
	return forest;
}

void run_detect(int argc, char** argv) {
	if (argc < 2) {
		cerr << "Usage: ./RandomForest --mode=detect <path-to-video>" << endl;
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

    vector<Mat> segmentation(levelList.size());
    for(auto i=0; i<segmentation.size(); ++i)
        video_segment::segment_video(images, segmentation[i], levelList[i]);

    //empty ground truth
	cv::Ptr<ml::DTrees> classifier;
	if (FLAGS_classifier == "rf") {
		classifier = ml::RTrees::load<ml::RTrees>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	} else if (FLAGS_classifier == "bt") {
		classifier = ml::Boost::load<ml::Boost>(FLAGS_model);
		CHECK(classifier.get()) << "Can not load random forest: " << FLAGS_model;
	}
	printf("Max tree depth: %d\n", classifier->getMaxDepth());


    Mat refImage = images[0].clone();

    Mat detection;
    ML::detectImage(images, segmentation, classifier, detection);

    Mat mask(detection.size(), CV_8UC3, Scalar(255,0,0));
    for(auto y=0; y<detection.rows; ++y){
        for(auto x=0; x<detection.cols; ++x){
            if(detection.at<uchar>(y,x) > (uchar)200)
                mask.at<Vec3b>(y,x) = Vec3b(0,0,255);
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
