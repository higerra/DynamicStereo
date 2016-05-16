//
// Created by yanhang on 5/15/16.
//

#include "tempMeanshift.h"
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
            cvtColor(frame, frame, CV_BGR2Luv);
            const uchar* pFrame = frame.data;
            for(auto i=0; i<width * height; ++i){
                array[i][fid*3] = pFrame[3*i];
                array[i][fid*3+1] = pFrame[3*i+1];
                array[i][fid*3+2] = pFrame[3*i+2];
            }
        }

	    return dsize;
    }

	cv::Mat segment(const std::vector<std::vector<float> >& array, const cv::Size& size){
		meanshift::MeanShift ms;
		//define data

	}

}//namespace dynamic_stereo