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

	char buffer[128] = {};
	TrainDataGUI gui;

	if(FLAGS_video){

	}else{
		int index = 0;
		while(true){
			sprintf(buffer, "%s/image%05d.jpg", argv[1], index);
			Mat img = imread(buffer);
			if(!img.data)
				break;

			if(!gui.processImage(img))
				break;
			index++;
		}
	}

	return 0;
}