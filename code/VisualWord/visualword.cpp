//
// Created by yanhang on 7/19/16.
//

#include "visualword.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{
    void sampleKeyPoints(const std::vector<cv::Mat>& input, std::vector<cv::KeyPoint>& keypoints, const int sigma_s, const int sigma_r){
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;
        const int kFrame = (int)input.size();

        const int& rS = sigma_s;
        const int& rT = sigma_r;
        keypoints.reserve((size_t)(width / rS * height / rS * kFrame / rT));

        for(auto x=rS+1; x<width - rS; x += rS){
            for(auto y=rS+1; y<height - rS; y += rS){
                for(auto t = rT+1; t < kFrame - rT; t += rT){
                    cv::KeyPoint keypt;
                    keypt.pt = cv::Point2f(x,y);
                    keypt.octave = t;
                    keypoints.push_back(keypt);
                }
            }
        }
    }



    void writeCodebook(const std::string& path, const cv::Mat& codebook){
        ofstream fout(path.c_str());
        if(!fout.is_open()) {
            cerr << "Can not open file to write: " << path << endl;
            return;
        }
        CHECK_EQ(codebook.type(), CV_32FC1);
        fout << codebook.rows << ' ' << codebook.cols << endl;
        for(auto i=0; i<codebook.rows; ++i){
            for(auto j=0; j<codebook.cols; ++j){
                fout << codebook.at<float>(i,j) << ' ';
            }
            fout << endl;
        }
        fout.close();
    }

    bool loadCodebook(const std::string& path, cv::Mat& codebook){
        ifstream fin(path.c_str());
        if(!fin.is_open())
            return false;
        int row, col;
        fin >> row >> col;
        codebook.create(row, col, CV_32FC1);
        for(auto i=0; i<row; ++i){
            for(auto j=0; j<col; ++j)
                fin >> codebook.at<float>(i,j);
        }
        fin.close();
        return true;
    }

    double testClassifier(const cv::Ptr<cv::ml::TrainData> testPtr, const cv::Ptr<cv::ml::StatModel> classifier){
        CHECK(testPtr.get());
        CHECK(classifier.get());

        Mat result;
        classifier->predict(testPtr->getSamples(), result);
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