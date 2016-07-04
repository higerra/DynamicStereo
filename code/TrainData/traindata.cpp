//
// Created by Yan Hang on 6/24/16.
//

#include "traindata.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	TrainDataGUI::TrainDataGUI(const int kNeg_, const std::string wname):
			kNeg(kNeg_), sizeNeg(100), drag(false), sample_pos(true), lastPoint(-1,-1), offsetPos(0), offsetNeg(0),
			window_handler(wname), paintRect(0,0,-1,-1), max_width(1280){
	}

	void TrainDataGUI::printHelp() {

	}

	bool TrainDataGUI::processImage(const cv::Mat &baseImg, std::vector<cv::Rect>& pos, std::vector<cv::Rect>& neg) {
		reset();
		image = baseImg.clone();
		sizeNeg = image.cols / 20;
		if(image.cols > max_width){
			pyrDown(image, image);
			downsample = 2.0;
		}
		bool ret = eventLoop();
		posSample.swap(pos);
		negSample.swap(neg);
		return ret;
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
				}else if(!negSample.empty()){
					negSample.pop_back();
				}
				drag = false;
			}
			render();
		}
		destroyWindow(window_handler);
		return ret;
	}

	void TrainDataGUI::randomNegativeSample() {
		unsigned int seed = (unsigned int)time(NULL);
		cv::RNG rng(seed);
		while((int)negSample.size() < kNeg){
			Rect newRect;
			while(true){
				int x = rng.uniform(0, static_cast<int>(image.cols * downsample- sizeNeg));
				int y = rng.uniform(0, static_cast<int>(image.rows * downsample - sizeNeg));
				newRect = cv::Rect(cv::Point(x,y), cv::Size(sizeNeg, sizeNeg));

				bool nointersect = true;
				for(const auto& rec: negSample){
					if(rectIntersect(newRect, rec)){
						nointersect = false;
						break;
					}
				}
				if(!nointersect)
					continue;
				for(const auto& rec: posSample){
					if(rectIntersect(newRect, rec)){
						nointersect = false;
						break;
					}
				}
				if(nointersect)
					break;
			}
			negSample.push_back(newRect);
		}
	}

	void mouseFunc(int event, int x, int y, int, void *data) {
		TrainDataGUI *gui = (TrainDataGUI* ) data;
		if(event == cv::EVENT_LBUTTONDOWN && ! gui->drag){
			gui->lastPoint = cv::Point(x,y);
			gui->drag = true;
			gui->render();
		}else if(event == cv::EVENT_LBUTTONUP){
			if(gui->drag){
				cv::Rect curRect(gui->lastPoint * gui->downsample, cv::Point(x,y) * gui->downsample);
				if(gui->sample_pos) {
					gui->posSample.push_back(curRect);
				}
				else {
					gui->negSample.push_back(curRect);
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
			cv::rectangle(paintImg, posSample[i] / downsample, Scalar(0,255,0));
		}
		for(auto i=offsetNeg; i<negSample.size(); ++i){
			cv::rectangle(paintImg, negSample[i] / downsample, Scalar(0,0,255));
		}
		if(drag && paintRect.width > 0 && paintRect.height > 0){
			cv::rectangle(paintImg, paintRect, Scalar(255,255,0));
		}
		char buffer[64] = {};
		sprintf(buffer, "num of pos: %d, num of neg: %d", (int)posSample.size(), (int)negSample.size());

		cv::putText(paintImg, string(buffer), cv::Point(20,20), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255,0));
		if(sample_pos)
			cv::putText(paintImg, "Positive", cv::Point(20,35), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
		else
			cv::putText(paintImg, "Negative", cv::Point(20,35), cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
		imshow(window_handler, paintImg);
	}

	void saveTrainingSet(const std::string& path, const std::vector<TrainFile>& samples){
		CHECK(!samples.empty()) << "Empty sample set";
		char buffer[1024] = {};
		for(const auto& sample: samples){
			sprintf(buffer, "%s/%s.sample.txt", path.c_str(), sample.filename.c_str());
			ofstream fout(buffer);
			CHECK(fout.is_open()) << buffer;
			fout << sample.posSample.size() << ' ' << sample.negSample.size() << endl;
			for(const auto& pos: sample.posSample)
				fout << pos.x << ' ' << pos.y << ' ' << pos.width << ' ' << pos.height << endl;
			for(const auto& neg: sample.negSample)
				fout << neg.x << ' ' << neg.y << ' ' << neg.width << ' ' << neg.height << endl;
			fout.close();
		}
	}


}//namespace dynamic_stereo