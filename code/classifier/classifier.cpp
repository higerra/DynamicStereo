//
// Created by yanhang on 5/19/16.
//

#include "classifier.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void trainSVM(const string& input_path, const string& output_path){
        Ptr<ml::SVM> svm = ml::SVM::create();
        Ptr<ml::TrainData> trainData = ml::TrainData::loadFromCSV(input_path, 1);
        CHECK(trainData.get()) << "Can not load training data: " << input_path;
//        ml::ParamGrid gridC(pow(2.0,-3), pow(2.0,3), 2);
//        ml::ParamGrid gridG(pow(2.0,-3), pow(2.0,3), 2);
//        ml::ParamGrid emptyGrid(0,0,0);
//
	    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, FLT_EPSILON));
//        svm->trainAuto(trainData, 5);

	    svm->setType(ml::SVM::C_SVC);
	    svm->setKernel(ml::SVM::RBF);
        svm->setC(1.0);
        svm->setGamma(4.0);

        svm->train(trainData);

        Mat predictLabel;
        Mat gtLabel = trainData->getResponses();
        svm->predict(trainData->getSamples(), predictLabel);
        float acc = 0.0;
	    float kPos = 0.0;
        for(auto i=0; i<trainData->getNSamples(); ++i){
	        CHECK(predictLabel.at<float>(i,0) < 0.001 || predictLabel.at<float>(i,0) > 0.999) << predictLabel.at<float>(i,0);

            float diff = gtLabel.at<float>(i,0) - predictLabel.at<float>(i,0);
            if(abs(diff) < 0.1)
	            acc += 1.0;
	        if(gtLabel.at<float>(i,0) > 0.9)
		        kPos += 1.0;
        }
        printf("Ratio of positive samples: %.3f, Training accuracy: %.3f\n", kPos / (float)trainData->getNSamples(), acc / (float)trainData->getNSamples());
        svm->save(output_path);
    }

    Mat predictSVM(const string& model_path, const string& data_path, const int width, const int height){
        //Ptr<ml::SVM> svm = ml::SVM::load(model_path);
        Ptr<ml::SVM> svm = ml::SVM::load<ml::SVM>(model_path);
        CHECK(svm.get()) << "Can not load trained model: " << model_path;
        Ptr<ml::TrainData> testData = ml::TrainData::loadFromCSV(data_path, 1);
        CHECK(testData.get()) << "Can not load test data: " << data_path;
        const int N = testData->getNSamples();
        const int dim = testData->getNVars();
        printf("Number of test samples: %d, number of features: %d\n", N, dim);
        CHECK_EQ(testData->getNSamples(), width * height);

        Mat samples = testData->getSamples();
        Mat result;
        svm->predict(samples, result);
        CHECK_EQ(result.rows, width * height);
        const float* pLabels = (float*) result.data;
        Mat visualize(height, width, CV_8UC3, Scalar(0,0,0));
        for(auto i=0; i<width * height; ++i){
            if(pLabels[i] < 0.1)
                visualize.at<Vec3b>(i/width, i%width) = Vec3b(0,0,0);
            else if(pLabels[i] > 0.9)
                visualize.at<Vec3b>(i/width, i%width) = Vec3b(255,255,255);
            else
                CHECK(true) << "Unrecognized label: " << pLabels[i];
        }

        return visualize;
    }

}//namespace dynamic_stereo
