//
// Created by yanhang on 5/15/16.
//

#include "extracfeature.h"
#include "regiondescriptor.h"
#include "../external/segment_gb/segment-image.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    void DataSet::dumpData_libsvm(const std::string &path) const {
        if(features.empty()) {
            cerr << "Empty dataset" << endl;
            return;
        }
        ofstream fout(path.c_str());
        CHECK(fout.is_open());
        char buffer[1024] = {};
        for(auto i=0; i<features.size(); ++i){
            for(auto j=0; j<features[i].size(); ++j){
                fout << i;
                for(auto k=0; k<features[i][j].size(); ++k){
                    if(features[i][j][k] > 0){
                        sprintf(buffer, " %d:%.3f", k, features[i][j][k]);
                        fout << buffer;
                    }
                }
                fout << endl;
            }
        }
        fout.close();
    }

    void DataSet::dumpData_csv(const std::string &path) const {
        if(features.empty()) {
            cerr << "Empty dataset" << endl;
            return;
        }
        ofstream fout(path.c_str());
        CHECK(fout.is_open());
        //header
        for(auto i=0; i<features[0][0].size(); ++i)
            fout << 'v' << i << ',';
        fout << "label" << endl;
        for(auto i=0; i<features.size(); ++i){
            for(auto j=0; j<features[i].size(); ++j){
                for(auto k=0; k<features[i][j].size(); ++k)
                    fout << features[i][j][k] << ',';
                fout << i << endl;
            }
        }
        fout.close();
    }

    void DataSet::appendDataSet(const DataSet& newData){
        const FeatureSet& newFeature = newData.getFeatures();
        if(features.empty())
            features.resize(newFeature.size());
        CHECK_EQ(features.size(), newFeature.size());
        for(auto i=0; i<features.size(); ++i)
            features[i].insert(features[i].end(), newFeature[i].begin(), newFeature[i].end());
    }

    namespace Feature {
        cv::Size importData(const std::string& path, std::vector<std::vector<float> >& array, const int downsample, const int tWindow,
                            const bool contain_invalid){
            VideoCapture cap(path);
            CHECK(cap.isOpened());
            int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) / downsample;
            int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) / downsample;
	        const int kFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);

	        int kNum;
	        if(tWindow > 0 && tWindow <= kFrame)
		        kNum = tWindow;
	        else
		        kNum = kFrame;

            array.resize(width * height);
            for(auto& a: array)
                a.reserve(kFrame * 3);

            const int nLevel = (int)std::log2((double)downsample) + 1;

            cv::Size dsize(width, height);

            for(auto fid=0; fid<kNum; ++fid){
                Mat frame;
                if(!cap.read(frame)) {
	                break;
                }
                vector<Mat> pyramid(nLevel);
                pyramid[0] = frame.clone();
                for(auto l=1; l<nLevel; ++l)
                    cv::pyrDown(pyramid[l-1],pyramid[l]);
                CHECK_EQ(pyramid.back().cols, width);
                CHECK_EQ(pyramid.back().rows, height);
                //cvtColor(frame, frame, CV_BGR2Luv);
                cvtColor(frame, frame, CV_BGR2RGB);
                const uchar* pFrame = pyramid.back().data;
                for(auto i=0; i<width * height; ++i){
                    Vector3f curpix((float)pFrame[3*i], (float)pFrame[3*i+1], (float)pFrame[3*i+2]);
//                    if(contain_invalid && curpix.norm() < 0.1)
//                        continue;
                    array[i].push_back(curpix[0]);
                    array[i].push_back(curpix[1]);
                    array[i].push_back(curpix[2]);
                }
            }

            return dsize;
        }

        cv::Size importDataMat(const std::string& path, std::vector<cv::Mat>& output, const int downsample, const int tWindow){
            VideoCapture cap(path);
            CHECK(cap.isOpened());
            int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) / downsample;
            int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) / downsample;
            const int kFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);

            int kNum;
            if(tWindow > 0 && tWindow <= kFrame)
                kNum = tWindow;
            else
                kNum = kFrame;

            cv::Size dsize(width, height);

            const int nLevel = (int)std::log2((double)downsample) + 1;
            for(auto fid=0; fid<kNum; ++fid){
                Mat frame;
                if(!cap.read(frame)) {
                    break;
                }
                vector<Mat> pyramid(nLevel);
                pyramid[0] = frame.clone();
                for(auto l=1; l<nLevel; ++l)
                    cv::pyrDown(pyramid[l-1],pyramid[l]);
                CHECK_EQ(pyramid.back().cols, width);
                CHECK_EQ(pyramid.back().rows, height);
                //cvtColor(frame, frame, CV_BGR2Luv);
                cvtColor(pyramid.back(), pyramid.back(), CV_BGR2RGB);
                output.push_back(pyramid.back());
            }

            return dsize;
        }


        void extractFeature(const std::vector<std::vector<float> > &array, const cv::Size &dims, const cv::Mat &gt, DataSet& dataset,
                            const int kBin, const float min_diff, const FeatureType method) {
            CHECK(!array.empty());
            vector<vector<int> > pixInds;
            const int negStride = 2;
            const int width = dims.width;
            const int height = dims.height;

            DataSet::FeatureSet& samples = dataset.getFeatures();

            if (gt.data) {
                CHECK_EQ(gt.size(), dims);
                pixInds.resize(2);
                for (auto y = 0; y < height; y += negStride) {
                    for (auto x = 0; x < width; x += negStride) {
                        if (gt.at<uchar>(y, x) < (uchar)10)
                            pixInds[0].push_back(y * width + x);
                    }
                }
                for (auto y = 0; y < height; ++y) {
                    for (auto x = 0; x < width; ++x) {
                        if (gt.at<uchar>(y, x) > (uchar)200)
                            pixInds[1].push_back(y * width + x);
                    }
                }
                samples.resize(2);
                printf("kPositive: %d, kNegative: %d\n", (int)pixInds[1].size(), (int)pixInds[0].size());
            } else {
                pixInds.resize(1);
                for (auto id = 0; id < width * height; ++id)
                    pixInds[0].push_back(id);
                samples.resize(1);
            }

            //feature constructor
            shared_ptr<FeatureConstructor> featureConstructor(NULL);
            switch (method){
                case RGB_CAT:
                    featureConstructor.reset(new RGBHist(kBin, min_diff));
                    break;
                default:
                    CHECK(true) << "Unsupported method";
            }

            for (auto i = 0; i < samples.size(); ++i) {
                samples[i].resize(pixInds[i].size());
                for(auto pid=0; pid<pixInds[i].size(); ++pid) {
                    featureConstructor->constructFeature(array[pixInds[i][pid]], samples[i][pid]);
                }
            }
        }

        void extractTrainRegionFeature(const std::vector<cv::Mat>& images, const cv::Mat& gt, DataSet& samples){
            CHECK(!images.empty());

        }

        void extractTestRegionFeature(const std::vector<cv::Mat>& images, DataSet& samples){
            CHECK(!images.empty());

        }

    }//namespace Feature



}//namespace dynamic_stereo