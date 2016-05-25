//
// Created by yanhang on 5/19/16.
//

#include "classifier.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void train(const string& input_path, const string& output_path, const string& type){
        Ptr<ml::TrainData> trainData = ml::TrainData::loadFromCSV(input_path, 1);
        CHECK(trainData.get()) << "Can not load training data: " << input_path;
        Ptr<ml::StatModel> classifier;
        if(type == "SVM") {
            printf("Training SVM...\n");
            classifier = ml::SVM::create();
            cv::Ptr<ml::SVM> svm = classifier.dynamicCast<ml::SVM>();
            svm->setTermCriteria(
                    cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20000, FLT_EPSILON));

            svm->setType(ml::SVM::C_SVC);
            svm->setKernel(ml::SVM::RBF);
            svm->setC(1);
            svm->setGamma(4);
        }else if(type == "GBT"){
            printf("Training GBT...\n");
            classifier = ml::Boost::create();
            ml::Boost* gbt = dynamic_pointer_cast<ml::Boost>(classifier.get());
        }else{
            CHECK(true) << "Unsupported classifier type " << type;
        }

        classifier->train(trainData);

        Mat predictLabel;
        Mat gtLabel = trainData->getResponses();
        classifier->predict(trainData->getSamples(), predictLabel);
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
        classifier->save(output_path);
    }

    Mat predict(const string& model_path, const string& data_path, const int width, const int height, const string& type){
        //Ptr<ml::SVM> svm = ml::SVM::load(model_path);
        Ptr<ml::TrainData> testData = ml::TrainData::loadFromCSV(data_path, 1);
        CHECK(testData.get()) << "Can not load test data: " << data_path;
        const int N = testData->getNSamples();
        const int dim = testData->getNVars();
        printf("Number of test samples: %d, number of features: %d\n", N, dim);
        CHECK_EQ(testData->getNSamples(), width * height);
        Ptr<ml::StatModel> classifier;
        if(type == "SVM") {
            printf("Predicing with SVM...\n");
            classifier = ml::SVM::load<ml::SVM>(model_path);
        }else if(type == "GBT"){
            printf("Predicting with GBT...\n");
            classifier = ml::Boost::load<ml::Boost>(model_path);
        }else
            CHECK(true) << "Unsupported classifier type: " << type;

        Mat samples = testData->getSamples();
        Mat result;
        classifier->predict(samples, result);
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
