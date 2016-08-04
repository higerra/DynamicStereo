//
// Created by yanhang on 8/3/16.
//

#include "mlutility.h"
#include <fstream>

using namespace std;
using namespace cv;

namespace dynamic_stereo {
    namespace ML {
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

            cv::Ptr<cv::ml::TrainData> convertTrainData(const TrainSet &trainset) {
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

            void computeGradient(const cv::Mat &image, cv::Mat &gradient) {
                const float pi = 3.1415926f;
                Mat gx, gy, gray;
                cvtColor(image, gray, CV_BGR2GRAY);
                cv::Sobel(gray, gx, CV_32F, 1, 0);
                cv::Sobel(gray, gy, CV_32F, 0, 1);
                gradient.create(image.size(), CV_32FC2);
                for (auto y = 0; y < gx.rows; ++y) {
                    for (auto x = 0; x < gx.cols; ++x) {
                        float ix = gx.at<float>(y, x);
                        float iy = gy.at<float>(y, x);
                        Vec2f pix;
                        pix[0] = std::sqrt(ix * ix + iy * iy);
                        float tx = ix + std::copysign(0.000001f, ix);
                        //normalize atan value to [0,80PI]
                        pix[1] = (atan(iy / tx) + pi / 2.0f) * 80;
                        gradient.at<Vec2f>(y, x) = pix;
                    }
                }
            }

            void compute3DGradient(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& gradient){
                CHECK_GE(input.size(), 2);
                gradient.resize(input.size());
                vector<cv::Mat> grays(input.size());
                for(auto v=0; v<input.size(); ++v) {
                    cvtColor(input[v], grays[v], CV_BGR2GRAY);
                }
                for(auto v=0; v<input.size(); ++v) {
                    gradient[v].create(input[v].size(), CV_32FC3);
                    vector<cv::Mat> curG(3);
                    cv::Sobel(grays[v], curG[0], CV_32F, 1, 0);
                    cv::Sobel(grays[v], curG[1], CV_32F, 0, 1);
                    curG[2].create(input[v].size(), CV_32FC1);
                    for (auto y = 0; y < input[v].rows; ++y) {
                        for (auto x = 0; x < input[v].cols; ++x) {
                            if (v == 0) {
                                curG[2].at<float>(y, x) =
                                        (float) grays[v + 1].at<uchar>(y, x) - (float) grays[v].at<uchar>(y, x);
                            } else if (v == input.size() - 1) {
                                curG[2].at<float>(y, x) =
                                        (float) grays[v].at<uchar>(y, x) - (float) grays[v - 1].at<uchar>(y, x);
                            } else {
                                curG[2].at<float>(y, x) =
                                        ((float) grays[v + 1].at<uchar>(y, x) - (float) grays[v - 1].at<uchar>(y, x)) / 2;
                            }
                        }
                    }
                    cv::merge(curG, gradient[v]);
                }
            }

        }//namespace MLUtility
    }//namespace ML

}//namespace dynamic_stereo