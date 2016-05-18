//
// Created by yanhang on 5/15/16.
//

#include "extracfeature.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    void FeatureConstructor::normalizel2(std::vector<float> &array) const {
        const float epsilon = 1e-3;
        float sqsum = 0.0;
        for(auto f: array)
            sqsum += f * f;
        if(sqsum < epsilon)
            return;
        for(auto& f: array)
            f /= std::sqrt(sqsum);
    }

    void RGBCat::constructFeature(const std::vector<float>& array, std::vector<float>& feat) const{
        CHECK_EQ((int)array.size() % 3, 0);
        feat.resize((size_t)kBin * 3 + kBinIntensity, 0.0f);
        vector<float> feat_diff((size_t)kBin*3, 0.0f);
        vector<float> feat_intensity((size_t)kBinIntensity, 0.0f);

        Vector3f RGB2Gray(0.299, 0.587, 0.114);
        for(auto t=0; t<array.size()/3 - 1; ++t){
            Vector3f pix1(array[t*3],array[t*3+1],array[t*3+2]);
            Vector3f pix2(array[(t+1)*3],array[(t+1)*3+1],array[(t+1)*3+2]);

            //intensity
            float intensity = pix1.dot(RGB2Gray);
            int bidInt = floor(intensity / 256);
            CHECK_LT(bidInt, kBinIntensity);
            feat_intensity[bidInt] += 1.0;

            //color change
            Vector3f diff = pix2 - pix1;
            if(diff.norm() >= min_diff) {
                for (auto c = 0; c < 3; ++c) {
                    int bid = floor((diff[c] + 256) / binUnit);
                    CHECK_LT(kBin * c + bid, feat.size());
                    feat_diff[kBin * c + bid] += 1.0;
                }
            }
        }
        //normalize, cut and renormalize
        normalizel2(feat_intensity);
        normalizel2(feat_diff);
        for(auto& f: feat_intensity){
            if(f < cut_thres)
                f = 0;
        }
        for(auto& f: feat_diff){
            if(f < cut_thres)
                f = 0;
        }
        normalizel2(feat_intensity);
        normalizel2(feat_diff);

        feat.insert(feat.end(), feat_diff.begin(), feat_diff.end());
        feat.insert(feat.end(), feat_intensity.begin(), feat_intensity.end());
    }

    void DataSet::dumpData_libsvm(const std::string &path) const {
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

    void DataSet::appendDataSet(const DataSet& newData){
        const FeatureSet& newFeature = newData.getFeatures();
        if(features.empty())
            features.resize(newFeature.size());
        CHECK_EQ(features.size(), newFeature.size());
        for(auto i=0; i<features.size(); ++i)
            features[i].insert(features[i].end(), newFeature[i].begin(), newFeature[i].end());
    }

    namespace Feature {
        cv::Size importData(const std::string& path, std::vector<std::vector<float> >& array, const int downsample, const int tWindow){
            VideoCapture cap(path);
            CHECK(cap.isOpened());
            int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) / downsample;
            int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) / downsample;
            array.resize(width * height);
            for(auto& a: array)
                a.resize(tWindow * 3);

            cv::Size dsize(width, height);

            const int stride = width * height * 3;
            for(auto fid=0; fid<tWindow; ++fid){
                Mat frame;
                if(cap.read(frame))
                    break;
                cv::resize(frame, frame, dsize);
                //cvtColor(frame, frame, CV_BGR2Luv);
                cvtColor(frame, frame, CV_BGR2RGB);
                const uchar* pFrame = frame.data;
                for(auto i=0; i<width * height; ++i){
                    array[i][fid*3] = pFrame[3*i];
                    array[i][fid*3+1] = pFrame[3*i+1];
                    array[i][fid*3+2] = pFrame[3*i+2];
                }
            }

            return dsize;
        }


        void extractFeature(const std::vector<std::vector<float> > &array, const cv::Size &dims, const cv::Mat &gt, DataSet& dataset,
                            const int kBin, const float min_diff, const FeatureType method) {
            CHECK(!array.empty());
            vector<vector<int> > pixInds;
            const int negStride = 4;
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
                    featureConstructor.reset(new RGBCat(kBin, min_diff));
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

    }



}//namespace dynamic_stereo