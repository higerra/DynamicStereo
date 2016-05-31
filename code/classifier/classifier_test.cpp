//
// Created by yanhang on 5/30/16.
//

#include <gtest/gtest.h>
#include "classifier.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

//TEST(classifier, perturbSamples){
//	printf("Testing for perturb sample:\n");
//    Mat toyMat(5,5,CV_32F,Scalar::all(0));
//    cv::randu(toyMat, 0, 1);
//    cout << "Original mat:" << endl << toyMat << endl;
//    perturbSamples(toyMat);
//    cout << "After pertub:" << endl << toyMat << endl;
//}

//TEST(classifier, splitSamples) {
//	printf("Testing for splitSample():\n");
//	Mat toySample(10, 3, CV_32F, Scalar::all(0));
//	Mat toyResponse(10, 1, CV_32F, Scalar::all(0));
//
//	cv::randu(toySample, 0, 1);
//	cv::randu(toyResponse, 0, 1);
//
//	Mat toyOriAll;
//	cv::hconcat(toySample, toyResponse, toyOriAll);
//	cout << "Original data:" << endl << toyOriAll << endl;
//	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(toySample, cv::ml::ROW_SAMPLE, toyResponse);
//	const int kFold = 3;
//	vector<Mat> sptSample, sptResponse;
//	splitSamples(trainData, sptSample, sptResponse, kFold);
//	for (auto i = 0; i < kFold; ++i) {
//		Mat sptAll;
//		hconcat(sptSample[i], sptResponse[i], sptAll);
//		cout << "Split set " << i << endl << sptAll << endl;
//	}
//}

//TEST(classifier, svm_raw_output){
//	const int nSample = 20;
//
//	Mat ranSample(nSample,3,CV_32F, Scalar::all(0));
//	Mat ranResponse(nSample, 1, CV_32S, Scalar::all(0));
//	cv::randu(ranSample, 0, 1);
//	for(auto i=0; i<nSample/2; ++i){
//		ranResponse.at<int>(i, 0) = 0;
//		ranResponse.at<int>(i+nSample/2, 0) = 1;
//	}
//	cv::Ptr<ml::TrainData> toyTrainData = cv::ml::TrainData::create(ranSample, cv::ml::ROW_SAMPLE, ranResponse);
//	cv::Ptr<ml::SVM> svm = ml::SVM::create();
//	svm->train(toyTrainData);
//
//	Mat margin, label;
//	svm->predict(ranSample, label);
//	svm->predict(ranSample, margin, ml::StatModel::RAW_OUTPUT);
//	ASSERT_EQ(margin.rows, nSample);
//	ASSERT_EQ(label.rows, nSample);
//	ASSERT_EQ(margin.type(), CV_32F);
//	ASSERT_EQ(label.type(), CV_32F);
//
//	printf("Margin\tLabel\n");
//	for(auto i=0; i<nSample; ++i){
//		printf("%.3f\t%.1f\n", margin.at<float>(i,0), label.at<float>(i,0));
//	}
//
//}

TEST(classifier, opencv_vconcat){
	const int kFold = 3;
	vector<Mat> spt((size_t)kFold);
	for(auto i =0; i<kFold; ++i) {
		spt[i] = Mat(5,3,CV_32F, Scalar::all(i));
	}
	Mat largeMat(0, 3, CV_32F);
	for(auto i=0; i<3; ++i) {
		cout << "Mat " << i << endl << spt[i] << endl;
		cv::vconcat(largeMat, spt[i], largeMat);
	}
	cout << "Large mat: " << endl << largeMat << endl;
}

TEST(classifier, calc_sigmond){
}