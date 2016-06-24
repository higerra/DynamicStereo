//
// Created by Yan Hang on 6/24/16.
//

#include "traindata.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	void TrainDataGUI::printHelp() {

	}

	void TrainDataGUI::eventLoop() {
		imshow(window_handler, image);
		cv::setMouseCallback(window_handler, mouseFunc);
		while(true){
			int key = waitKey(0);
			if(key & 255 == 27)
				break;
			char c = (char)key;
			if(c == 'p'){
				drag = false;
				sample_pos = true;
			}else if(c == 'n'){
				drag = false;
				sample_pos = false;
			}else if(c == 'r'){
				drag = false;
				sample_pos = false;
				randomNegativeSample();
			}
		}
	}

	void TrainDataGUI::randomNegativeSample() {

	}

	void TrainDataGUI::mouseFunc(int event, int x, int y, void *data) {
	}


}//namespace dynamic_stereo