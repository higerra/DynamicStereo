//
// Created by Yan Hang on 6/24/16.
//

#ifndef DYNAMICSTEREO_TRAINDATA_H
#define DYNAMICSTEREO_TRAINDATA_H

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace dynamic_stereo{

	struct TrainSample{
		TrainSample(const std::string& filename_, const cv::Rect& roi_, const bool positive_):
				filename(filename_), roi(roi_), positive(positive_){}
		TrainSample(){}

		std::string filename;
		cv::Rect roi;
		bool positive;
	};

	struct TrainingOption{
		std::string path;
		int patch_size;
		std::vector<float> scale;
		std::vector<float> ratio;
	};

	class TrainDataGUI{
	public:

		TrainDataGUI(const cv::Mat& image_, const int kNeg_ = 50,
		             const std::string wname = "TrainingSample"):
				image(image_), kNeg(kNeg_), drag(false), sample_pos(true), lastPoint(-1,-1), window_handler(wname){
			eventLoop();
		}

		~TrainDataGUI(){
			cv::destroyWindow(window_handler);
		}

		void printHelp();
		void eventLoop();

		inline const std::vector<TrainSample>& getPosSample(){return posSample;}
		inline const std::vector<TrainSample>& getNegSample(){return negSample;}
		inline const std::vector<cv::Mat>& getPosImage(){return posImage;}
		inline const std::vector<cv::Mat>& getNegImage(){return negImage;}
	private:
		void mouseFunc(int event, int x, int y, void* data);
		void randomNegativeSample();

		const cv::Mat& image;
		const int kNeg;

		bool drag;
		bool sample_pos;

		cv::Point lastPoint;

		std::string window_handler;

		std::vector<TrainSample> posSample;
		std::vector<TrainSample> negSample;
		std::vector<cv::Mat> posImage;
		std::vector<cv::Mat> negImage;
	};

	void saveTrainingSet(const std::string& path);

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TRAINDATA_H
