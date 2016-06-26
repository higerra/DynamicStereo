//
// Created by Yan Hang on 6/24/16.
//

#include "traindata.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_bool(video, false, "video input");

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage: ./TrainData <path-to-data>" << endl;
	}
	google::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);
	if(argc < 2){
		cerr << "Usage: ./TrainData <path-to-data>" << endl;
	}

	char buffer[128] = {};
	TrainDataGUI gui;

	vector<TrainFile> samples;
	if(FLAGS_video){
		sprintf(buffer, "%s/list.txt", argv[1]);
		ifstream listIn(buffer);
		CHECK(listIn.is_open()) << "Data directory must contain list.txt";
		string filename;
		int index = 0;
		while(listIn >> filename){
			string fullPath = string(argv[1]) + filename;
			VideoCapture cap(fullPath);
			printf("Sampling from %s\n", fullPath.c_str());
			if(!cap.isOpened()) {
				cerr << "Can not open " << fullPath << ", skip" << endl;
				continue;
			}
			Mat thumb;
			cap >> thumb;
			vector<cv::Rect> curPos;
			vector<cv::Rect> curNeg;
			bool ret = gui.processImage(thumb, curPos, curNeg);
			TrainFile curFile;
			curFile.filename = string(buffer);
			curFile.posSample.swap(curPos);
			curFile.negSample.swap(curNeg);
			samples.push_back(curFile);
			if(!ret)
				break;
			index++;
		}
	}else{
		int index = 0;
		while(true){
			sprintf(buffer, "%s/image%05d.jpg", argv[1], index);
			Mat img = imread(buffer);
			if(!img.data)
				break;
			vector<cv::Rect> curPos, curNeg;
			bool ret = gui.processImage(img, curPos, curNeg);
			TrainFile curFile;
			sprintf(buffer, "image%05d.jpg", index);
			curFile.filename = string(buffer);
			curFile.posSample.swap(curPos);
			curFile.negSample.swap(curNeg);
			samples.push_back(curFile);
			if(!ret)
				break;
			index++;
		}
	}
	sprintf(buffer, "%s/train_hough.txt", argv[1]);
	saveTrainingSet(string(buffer), samples);

	return 0;
}