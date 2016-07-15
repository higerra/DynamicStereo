//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"
#include "../base/utility.h"
#include "../common/descriptor.h"
#include <fstream>

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	cv::Ptr<cv::ml::TrainData> convertTrainData(const Feature::TrainSet& trainset){
		CHECK_EQ(trainset.size(), 2);
		CHECK(!trainset[0].empty());

		Mat sampleMat((int)trainset[0].size()+(int)trainset[1].size(), (int)trainset[0][0].feature.size(), CV_32FC1, Scalar::all(0));
		Mat response((int)trainset[0].size() + (int)trainset[1].size(), 1, CV_32S, Scalar::all(0));
		int index = 0;

		for(auto l=0; l<trainset.size(); ++l) {
			for (auto feat: trainset[l]) {
				for (auto i = 0; i < feat.feature.size(); ++i)
					sampleMat.at<float>(index, i) = feat.feature[i];
				response.at<int>(index, 0) = l;
				index++;
			}
		}

		cv::Ptr<cv::ml::TrainData> traindata = ml::TrainData::create(sampleMat, cv::ml::ROW_SAMPLE, response);
		return traindata;
	}

	void saveTrainData(const std::string& path, const Feature::TrainSet& trainset){
		CHECK(!trainset.empty());
		CHECK_LT(trainset.size(), 3);

		ofstream trainOut(path.c_str());
		CHECK(trainOut.is_open()) << path;

		for(auto l=0; l<trainset.size(); ++l){
			for(const auto& sample: trainset[l]){
				for(auto i=0; i<sample.feature.size(); ++i){
					trainOut << sample.feature[i] << ',';
				}
				trainOut << l << endl;
			}
		}

		trainOut.close();
	}

	double testForest(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::DTrees> forest){
		CHECK(testPtr.get());
		CHECK(forest.get());

		Mat result;
		forest->predict(testPtr->getSamples(), result);
		Mat groundTruth;
		testPtr->getResponses().convertTo(groundTruth, CV_32F);

		CHECK_EQ(groundTruth.rows, result.rows);
		float acc = 0.0f;
		for(auto i=0; i<result.rows; ++i){
			float gt = groundTruth.at<float>(i,0);
			float res = result.at<float>(i,0);
			if(std::abs(gt-res) <= 0.1)
				acc += 1.0f;
		}
		return acc / (float)result.rows;

	}
}//namespace dynamic_stereo