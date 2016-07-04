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
#include <fstream>

namespace dynamic_stereo{

	struct TrainFile{
		std::string filename;
		std::vector<cv::Rect> posSample;
		std::vector<cv::Rect> negSample;
	};

	inline cv::Rect operator * (const cv::Rect& rec, double ratio){
		cv::Rect tmp(rec.tl()*ratio, rec.br()*ratio);
		return tmp;
	}

	inline cv::Rect operator / (const cv::Rect& rec, double ratio){
		CHECK_GT(ratio, 0);
		cv::Rect tmp(rec.tl()/ratio, rec.br()/ratio);
		return tmp;
	}

	inline bool rectIntersect(const cv::Rect& rec1, const cv::Rect& rec2){
		cv::Rect inter = rec1 & rec2;
		return (inter.width > 0);
	}

	class TrainDataGUI{
	public:
		TrainDataGUI(const int kNeg_ = 20, const std::string wname = "TrainingSample");

		~TrainDataGUI(){
			cv::destroyWindow(window_handler);
		}

		bool processImage(const cv::Mat& baseImg, std::vector<cv::Rect>& pos, std::vector<cv::Rect>& neg);

		void printHelp();

		inline void reset(){
			posSample.clear();
			negSample.clear();
			drag = false;
			sample_pos = true;
			downsample = 1.0;
			offsetNeg = 0;
			offsetPos = 0;
			image.release();
		}
	private:
		void randomNegativeSample();
		bool eventLoop();
		void render();

		friend void mouseFunc(int event, int x, int y, int, void* data);

		cv::Mat image;
		cv::Mat paintImg;
		cv::Rect paintRect;

		int kNeg;
		int sizeNeg;

		int offsetPos;
		int offsetNeg;

		bool drag;
		bool sample_pos;

		double downsample;

		cv::Point lastPoint;

		std::string window_handler;

		const int max_width;

		std::vector<cv::Rect>posSample;
		std::vector<cv::Rect> negSample;
	};

	void mouseFunc(int event, int x, int y, int, void* data);
	void saveTrainingSet(const std::string& path, const std::vector<TrainFile>& trainSamples);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAINDATA_H
