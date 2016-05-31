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

    void splitSamples(const cv::Ptr<cv::ml::TrainData> input,
                      std::vector<cv::Mat>& outputSample, std::vector<cv::Mat>& outputLabel,
                      const int kFold){
        CHECK(input.get());
        int nSample = input->getNSamples();
        vector<int> crossNum(kFold, 0);
        vector<int> crossCount(kFold, 0);
        for(auto i=0; i<nSample; ++i)
            crossNum[i%kFold]++;
        outputSample.resize((size_t)kFold);
	    outputLabel.resize((size_t)kFold);
        for(auto i=0; i<kFold; ++i){
	        outputSample[i].create(crossNum[i], input->getNVars(), CV_32F);
	        outputLabel[i].create(crossNum[i], 1, CV_32S);
        }

	    Mat sampleMat = input->getSamples();
	    Mat responseMat = input->getResponses();
	    CHECK_EQ(responseMat.type(), CV_32F);
        for(auto i=0; i<nSample; ++i){
            int ind = i % kFold;
            CHECK_LT(crossCount[ind], outputLabel[ind].rows);
	        sampleMat.rowRange(i,i+1).copyTo(outputSample[ind].rowRange(crossCount[ind], crossCount[ind]+1));
	        //responseMat.rowRange(i,i+1).convertTo(outputLabel[ind].rowRange(crossCount[ind], crossCount[ind]+1), CV_32S);
	        outputLabel[ind].at<int>(crossCount[ind], 0) = (int)responseMat.at<float>(i,0);
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
            classifier = ml::SVM::load(model_path);
//            classifier = ml::SVM::load<ml::SVM>(model_path);
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
        vector<cv::Mat> sptSamples;
	    vector<cv::Mat> sptResponse;
        splitSamples(trainData, sptSamples, sptResponse, kFold);
        //trainData.release();

        Mat result(0,1,CV_32F);

        for(auto i=0; i<kFold; ++i) {
	        Ptr<ml::SVM> classifier = ml::SVM::create();

	        Mat trainSample(0, sptSamples[i].cols, CV_32F);
	        Mat trainResponse(0, 1, CV_32S, Scalar::all(0));
	        for(auto j=0; j<kFold; ++j){
		        if(j == i)
			        continue;
		        cv::vconcat(trainSample, sptSamples[j], trainSample);
		        cv::vconcat(trainResponse, sptResponse[j], trainResponse);
	        }

	        classifier->setC(1);
	        classifier->setGamma(4);
	        classifier->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, FLT_EPSILON));
	        printf("Training, fold %d/%d\n", i, kFold);
	        classifier->train(trainSample, ml::ROW_SAMPLE, trainResponse);

	        Mat margin;
//	        Mat label;
	        classifier->predict(sptSamples[i], margin, cv::ml::StatModel::RAW_OUTPUT);
//	        classifier->predict(sptSamples[i], label);
	        cv::vconcat(result, margin, result);
	        printf("Done. result.rows:%d\n", result.rows);
        }
	    printf("Training SVM...\n");
	    Ptr<ml::SVM> svm = ml::SVM::create();
	    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, FLT_EPSILON));
	    svm->setC(1);
	    svm->setGamma(4);
	    svm->train(trainData);

	    printf("Training Platt Scaling...\n");
	    cv::Ptr<ml::LogisticRegression> platt = trainPlattScaling(result);
	    CHECK(platt.get());
	    string platt_path = output_path + ".platt";
	    printf("Done\n. Saving svm as %s\n", output_path.c_str());
	    svm->save(output_path);
	    printf("Saving platt scaling as %s\n", platt_path.c_str());
	    platt->save(platt_path);
    }

	cv::Mat predictSVMWithPlatt(const std::string& model_path, const std::string& data_path, const int width, const int height){
		string platt_path = model_path + ".platt";
		cv::Ptr<ml::SVM> svm = ml::SVM::load(model_path);
		CHECK(svm.get()) << "Can not load SVM: " << model_path;
		cv::Ptr<ml::LogisticRegression> log = ml::LogisticRegression::load<ml::LogisticRegression>(platt_path);
		CHECK(svm.get()) << "Can not load Platt model: " << platt_path;

		cv::Ptr<ml::TrainData> testData = ml::TrainData::loadFromCSV(data_path, 1);
		CHECK(testData.get()) << "Can not test data: " << data_path;
		Mat samples = testData->getSamples();

		Mat result;
		log->predict(samples, result);
		CHECK_EQ(result.type(), CV_32F);
		const float* pResult = (float *)result.data;
		Mat visualize(height, width, CV_8UC3, Scalar(0,0,0));
		for(auto i=0; i<width * height; ++i){
			float tv = pResult[i];
			if(tv > 1.0 || tv < 0.0)
				CHECK(true) << "Unsupported label: " << pResult[i];
			tv *= 256;
			visualize.at<Vec3b>(i/width, i%width) = Vec3b((uchar)tv,(uchar)tv,(uchar)tv);
		}
		return visualize;
	}


	cv::Ptr<ml::LogisticRegression> trainPlattScaling(const cv::Mat& trainData){
		CHECK_EQ(trainData.type(), CV_32F);
		CHECK_EQ(trainData.cols, 1);
		CHECK_GT(trainData.rows, 0);

		const float* pData = (float*) trainData.data;
		float kPos = 0, kNeg = 0;
		for(auto i=0; i<trainData.rows; ++i){
			if(pData[i] < 0)
				kPos += 1.0;
			else
				kNeg += 1.0;
		}
		const float tPos = (kPos+1) / (kPos+2);
		const float tNeg = 1.0 / (kNeg + 2);

		Mat response(trainData.rows, 1, CV_32F, Scalar::all(0));
		float* pResponse = (float*) response.data;
		for(auto i=0; i<trainData.rows; ++i){
			if(pData[i] < 0)
				pResponse[i] = tPos;
			else
				pResponse[i] = tNeg;
		}

		cv::Ptr<ml::LogisticRegression> logReg = ml::LogisticRegression::create();
		logReg->train(trainData, ml::ROW_SAMPLE, response);
		return logReg;
	}

}//namespace dynamic_stereo
