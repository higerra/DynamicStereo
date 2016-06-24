//
// Created by Yan Hang on 6/24/16.
//

#include "traindata.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	TrainDataGUI::TrainDataGUI(const int kNeg_, const std::string wname):
			kNeg(kNeg_), drag(false), sample_pos(true), lastPoint(-1,-1), offsetPos(0), offsetNeg(0),
			window_handler(wname), paintRect(0,0,-1,-1){
	}

	void TrainDataGUI::printHelp() {

	}

	bool TrainDataGUI::processImage(const cv::Mat &baseImg) {
		image = baseImg.clone();
		return eventLoop();
	}

	bool TrainDataGUI::eventLoop() {
		namedWindow(window_handler);
		render();
		cv::setMouseCallback(window_handler, mouseFunc, this);
		bool ret = true;
		while(true){
			int key = waitKey(0);
			char c = (char)key;
			if(c == 'q') {
				ret = false;
				offsetPos = (int)posSample.size();
				offsetNeg = (int)negSample.size();
				break;
			}
			if(c == 'f') {
				offsetPos = (int)posSample.size();
				offsetNeg = (int)negSample.size();
				break;
			}
			if(c == 'p'){
				printf("Sample positive\n");
				drag = false;
				sample_pos = true;
			}else if(c == 'n'){
				printf("Sample negative\n");
				drag = false;
				sample_pos = false;
			}else if(c == 'r'){
				printf("Randomly generate negative sample\n");
				drag = false;
				sample_pos = false;
				randomNegativeSample();
			}else if(c == 'd'){
				printf("Sample deleted\n");
				if(sample_pos && (!posSample.empty())){
					posSample.pop_back();
					posImage.pop_back();
				}else if(!negSample.empty()){
					negSample.pop_back();
					negImage.pop_back();
				}
			}
			render();
		}
		destroyWindow(window_handler);
		return ret;
	}

	void TrainDataGUI::randomNegativeSample() {

	}

	void mouseFunc(int event, int x, int y, int, void *data) {
		TrainDataGUI *gui = (TrainDataGUI* ) data;
		if(event == cv::EVENT_LBUTTONDOWN && ! gui->drag){
			gui->lastPoint = cv::Point(x,y);
			gui->drag = true;
			gui->render();
		}else if(event == cv::EVENT_LBUTTONUP){
			if(gui->drag){
				cv::Rect curRect(gui->lastPoint, cv::Point(x,y));
				if(gui->sample_pos) {
					gui->posSample.push_back(curRect);
					gui->posImage.push_back(gui->image(curRect).clone());
				}
				else {
					gui->negSample.push_back(curRect);
					gui->negImage.push_back(gui->image(curRect).clone());
				}
				gui->drag = false;
			}
			gui->render();
		}else if(event == cv::EVENT_MOUSEMOVE){
			if(gui->drag) {
				gui->paintRect = cv::Rect(gui->lastPoint, cv::Point(x, y));
				gui->render();
			}
		}
	}

	void TrainDataGUI::render(){
		paintImg = image.clone();
		for(auto i=offsetPos; i<posSample.size(); ++i){
			cv::rectangle(paintImg, posSample[i], Scalar(0,255,0));
		}
		for(auto i=offsetNeg; i<negSample.size(); ++i){
			cv::rectangle(paintImg, negSample[i], Scalar(0,0,255));
		}
		if(drag && paintRect.width > 0 && paintRect.height > 0){
			cv::rectangle(paintImg, paintRect, Scalar(255,255,0));
		}
		char buffer[64] = {};
		sprintf(buffer, "num of pos: %d, num of neg: %d", (int)posImage.size(), (int)negImage.size());
		cv::putText(paintImg, string(buffer), cv::Point(20,20), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0,128,255));
		imshow(window_handler, paintImg);
	}

	void saveTrainingSet(const std::string& path){
		char buffer[128] = {};
		string posPath = path + "/posImage/";
		string negPath = path + "/negImage/";
		string posConfPath = path + "/train_pos.txt";
		string negConfPath = path + "/train_neg.txt";
	}


}//namespace dynamic_stereo