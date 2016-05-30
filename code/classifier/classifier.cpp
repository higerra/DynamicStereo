//
// Created by yanhang on 5/19/16.
//

#include "classifier.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void perturbSamples(cv::Mat& samples){
        CHECK(samples.data);
        const int nSample = samples.rows;
        Mat output(samples.rows, samples.cols, samples.type(), cv::Scalar::all(0));
        vector<int> shuffle(nSample);
        for(auto i=0; i<nSample; ++i)
            shuffle[i] = i;
        std::random_shuffle(shuffle.begin(), shuffle.end());
        for(auto i=0; i<nSample; ++i){
            samples.rowRange(shuffle[i], shuffle[i]+1).copyTo(output.rowRange(i, i+1));
        }
        cv::swap(samples, output);
    }

    void splitSamples(const cv::Ptr<cv::ml::TrainData> input, std::vector<cv::ml::TrainData>& output, const int kFold){
        CHECK(input.get());
        int nSample = input->getNSamples();
        vector<int> crossNum(kFold, 0);
        vector<int> crossCount(kFold, 0);
        for(auto i=0; i<nSample; ++i)
            crossNum[i%kFold]++;
        output.resize(kFold);
        for(auto i=0; i<kFold; ++i)
            output[i].create(crossNum[i], input.cols, input.type());
        for(auto i=0; i<input.rows; ++i){
            int ind = i % kFold;
            CHECK_LT(crossCount[ind], output[ind].rows);
            input.rowRange(i,i+1).copyTo(output[ind].rowRange(crossCount[ind], crossCount[ind]+1));
            crossCount[ind]++;
        }
    }

    void train(const string& input_path, const string& output_path, const string& type){
        //load training data, perturb the training data
        Ptr<ml::TrainData> trainData = ml::TrainData::loadFromCSV(input_path, 1);
        CHECK(trainData.get()) << "Can not load training data: " << input_path;
        //perturbSamples(trainData->getSamples());

        Ptr<ml::StatModel> classifier;
        if(type == "SVC" || type == "SVR") {
            classifier = ml::SVM::create();
            cv::Ptr<ml::SVM> svm = classifier.dynamicCast<ml::SVM>();
            svm->setTermCriteria(
                    cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, FLT_EPSILON));
            svm->setKernel(ml::SVM::RBF);
            svm->setC(1);
            if(type == "SVC") {
                printf("Training SVM classification...\n");
                svm->setType(ml::SVM::C_SVC);
                svm->setGamma(4);
            }else{
                printf("Training SVM regression...\n");
                svm->setType(ml::SVM::NU_SVR);
                svm->setNu(0.2);
            }
        } else if(type == "BT") {
            printf("Training Boosting Tree Regression...\n");
            classifier = ml::Boost::create();
            cv::Ptr<ml::Boost> bt = classifier.dynamicCast<ml::Boost>();
            bt->setBoostType(ml::Boost::REAL);
        }else{
            CHECK(true) << "Unsupported classifier type " << type;
        }

        classifier->train(trainData);

        //for binary classification, report training accuracy
        if(type == "SVC") {
            Mat predictLabel;
            Mat gtLabel = trainData->getResponses();
            classifier->predict(trainData->getSamples(), predictLabel);
            float acc = 0.0;
            float kPos = 0.0;
            for (auto i = 0; i < trainData->getNSamples(); ++i) {
                CHECK(predictLabel.at<float>(i, 0) < 0.001 || predictLabel.at<float>(i, 0) > 0.999)
                << predictLabel.at<float>(i, 0);
                float diff = gtLabel.at<float>(i, 0) - predictLabel.at<float>(i, 0);
                if (abs(diff) < 0.1)
                    acc += 1.0;
                if (gtLabel.at<float>(i, 0) > 0.9)
                    kPos += 1.0;
            }
            printf("Ratio of positive samples: %.3f, Training accuracy: %.3f\n",
                   kPos / (float) trainData->getNSamples(), acc / (float) trainData->getNSamples());
        }
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
        if(type == "SVC" || type == "SVR") {
            printf("Predicing with SVM...\n");
//            classifier = ml::SVM::load(model_path);
            classifier = ml::SVM::load<ml::SVM>(model_path);
        }else if(type == "BT"){
            printf("Predicting with GBT...\n");
            classifier = ml::Boost::load<ml::Boost>(model_path);
        }else
            CHECK(true) << "Unsupported classifier type: " << type;

        Mat samples = testData->getSamples();
        Mat result_label, result_dis;
        classifier->predict(samples, result_label);
        classifier->predict(samples, result_dis, cv::ml::StatModel::RAW_OUTPUT);

        CHECK_EQ(result_dis.rows, width * height);
        CHECK_EQ(result_label.rows, width * height);
        const float* pLabels = (float*) result_label.data;
        const float* pDis = (float*) result_dis.data;
        double minv, maxv;
        //get the max scale
        double scale = (maxv - minv);
        Mat visualize(height, width, CV_8UC3, Scalar(0,0,0));
        for(auto i=0; i<width * height; ++i){
            float dis = pDis[i];
            float label = pLabels[i];
            //tv = (tv-minv) / scale;
            float tv = pLabels[i];
            if(tv > 1.0 || tv < 0.0)
                CHECK(true) << "Unsupported label: " << pLabels[i];
            tv *= 256;
            visualize.at<Vec3b>(i/width, i%width) = Vec3b((uchar)tv,(uchar)tv,(uchar)tv);
        }

        return visualize;
    }

    void trainSVMWithPlatt(const std::string& input_path, const std::string& output_path){
        const int kFold = 3;
        Ptr<ml::TrainData> trainData = ml::TrainData::loadFromCSV(input_path, 1);
        CHECK(trainData.get()) << "Can not load training data: " << input_path;
        //perturbSamples(trainData->getSamples());
        Mat samples = trainData->getSamples();
        vector<cv::Mat> sptSamples;
        splitSamples(samples, sptSamples, kFold);
        samples.release();

        vector<Mat> result((size_t) kFold);

        for(auto i=0; i<kFold; ++i) {
            Ptr<ml::SVM> classifier = ml::SVM::create();
            classifier->setC(1);
            classifier->setGamma(4);
            Ptr<cv::ml::TrainData> tsample = cv::ml::TrainData::create()
        }
    }

}//namespace dynamic_stereo
