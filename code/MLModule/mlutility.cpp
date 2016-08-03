//
// Created by yanhang on 8/3/16.
//

#include "mlutility.h"
#include <fstream>

using namespace std;
using namespace cv;

namespace dynamic_stereo {
    namespace MLUtility {

        void normalizel2(std::vector<float> &array) {
            const float epsilon = 1e-3;
            float sqsum = 0.0;
            for (auto f: array)
                sqsum += f * f;
            if (sqsum < epsilon)
                return;
            for (auto &f: array)
                f /= std::sqrt(sqsum);
        }

        void normalizeSum(std::vector<float> &array) {
            const float epsilon = 1e-3;
            float sum = std::accumulate(array.begin(), array.end(), 0.0f);
            if (sum < epsilon)
                return;
            for (auto &f: array)
                f /= sum;
        }

        cv::Ptr<cv::ml::TrainData> convertTrainData(const Feature::TrainSet &trainset) {
            CHECK_EQ(trainset.size(), 2);
            CHECK(!trainset[0].empty());

            Mat sampleMat((int) trainset[0].size() + (int) trainset[1].size(), (int) trainset[0][0].feature.size(),
                          CV_32FC1, Scalar::all(0));
            Mat response((int) trainset[0].size() + (int) trainset[1].size(), 1, CV_32S, Scalar::all(0));
            int index = 0;

            for (auto l = 0; l < trainset.size(); ++l) {
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

        void writeTrainData(const std::string &path, const cv::Ptr<cv::ml::TrainData> traindata) {
            ofstream fout(path.c_str());
            if (!fout.is_open()) {
                cerr << "Can not open file to write: " << path << endl;
                return;
            }

            Mat feature = traindata->getSamples();
            Mat response = traindata->getResponses();
            CHECK_EQ(feature.type(), CV_32FC1);
            CHECK_EQ(response.type(), CV_32SC1);

            for (auto i = 0; i < feature.rows; ++i) {
                for (auto j = 0; j < feature.cols; ++j) {
                    fout << feature.at<float>(i, j) << ',';
                }
                fout << response.at<int>(i, 0) << endl;
            }

            fout.close();
        }
    }
}//namespace dynamic_stereo