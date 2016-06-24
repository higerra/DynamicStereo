//
// Created by Yan Hang on 6/24/16.
//

#ifndef DYNAMICSTEREO_TRAINDATA_H
#define DYNAMICSTEREO_TRAINDATA_H

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace dynamic_stereo{

	struct TrainUnit{
		std::string filename;
		std::vector<cv::Rect> posSample;
		std::vector<cv::Rect> negSample;
	};

	class TrainDataGUI{
	public:
		TrainDataGUI(const int kNeg_ = 50, const std::string wname = "TrainingSample");

		~TrainDataGUI(){
			cv::destroyWindow(window_handler);
		}

		bool processImage(const cv::Mat& baseImg);

		void printHelp();

		inline void reset(){
			posSample.clear();
			negSample.clear();
			posImage.clear();
			negImage.clear();
			drag = false;
			sample_pos = true;
			offsetNeg = 0;
			offsetPos = 0;
			image.release();
		}

		inline const std::vector<cv::Mat>& getPosImage(){return posImage;}
		inline const std::vector<cv::Mat>& getNegImage(){return negImage;}
	private:
		void randomNegativeSample();
		bool eventLoop();
		void render();

		friend void mouseFunc(int event, int x, int y, int, void* data);

		cv::Mat image;
		cv::Mat paintImg;
		cv::Rect paintRect;

		const int kNeg;
		int offsetPos;
		int offsetNeg;

		bool drag;
		bool sample_pos;

		cv::Point lastPoint;

		std::string window_handler;

		std::vector<cv::Rect>posSample;
		std::vector<cv::Rect> negSample;
		std::vector<cv::Mat> posImage;
		std::vector<cv::Mat> negImage;
	};

	void mouseFunc(int event, int x, int y, int, void* data);
	void saveTrainingSet(const std::string& path);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAINDATA_H
