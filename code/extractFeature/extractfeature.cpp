//
// Created by yanhang on 5/15/16.
//

#include "extracfeature.h"
#include "../external/segment_ms/ms.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{
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

    void extractFeatureRGBCat(const std::vector<std::vector<float> >& array, const cv::Size& dims, std::vector<std::vector<float> >& features, const int kBin){
        CHECK(!array.empty());
        features.resize(array.size());
        const float binUnit = 512 / (float)kBin;
        for(auto pid=0; pid < array.size(); ++pid){
            CHECK_EQ((int)array[pid].size() % 3, 0);
            features[pid].resize((size_t)kBin * 3);
            for(auto t=0; t<array[pid].size()/3 - 1; ++t){
                for(auto c=0; c<3; ++c){
                    float diff = array[pid][(t+1)*3+c] - array[pid][t*3+c];
                    int bid = floor((diff + 256) / binUnit);
                    CHECK_LT(bid, kBin);
                    features[pid][kBin*c+bid] += 1.0;
                }
            }
            //normalize
            float sqsum = 0.0;
            for(auto bid=0; bid<features[pid].size(); ++bid)
                sqsum += features[pid][bid] * features[pid][bid];
            CHECK_GT(sqsum, 0.1);
            for (auto bid = 0; bid < features[pid].size(); ++bid)
                features[pid][bid] /= sqsum;
        }
    }

}//namespace dynamic_stereo