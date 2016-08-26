//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"
#include "../base/utility.h"
#include <fstream>

using namespace std;
using namespace cv;

namespace dynamic_stereo{

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