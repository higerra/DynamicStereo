//
// Created by yanhang on 4/7/16.
//

#include "dynamic_confidence.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>

using namespace std;
using namespace dynamic_stereo;
using namespace cv;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 0, "tWindow");
int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./DynamicConfidence <path-to-directory>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    FileIO file_io(argv[1]);
    if(!stlplus::folder_exists(file_io.getDirectory() + "/opticalflow"))
        stlplus::folder_create(file_io.getDirectory() + "/opticalflow");
    if(!stlplus::folder_exists(file_io.getDirectory() + "/dynamic_confidence"))
        stlplus::folder_create(file_io.getDirectory() + "/dynamic_confidence");

    char buffer[1024] = {};
    try{
	    cout << "Checking optical flow..." << endl;
        int startid = FLAGS_testFrame - FLAGS_tWindow / 2;
        int endid = FLAGS_testFrame + FLAGS_tWindow / 2;
        CHECK_GE(startid, 0);
        CHECK_LT(endid, file_io.getTotalNum());
        bool flowMissing = false;
        for(auto i=startid; i<=endid; ++i){
            if(i < file_io.getTotalNum() - 1) {
                if (!stlplus::file_exists(file_io.getOpticalFlow_forward(i))) {
                    flowMissing = true;
                    break;
                }
            }
            if(i > 0) {
                if (!stlplus::file_exists(file_io.getOpticalFlow_backward(i))) {
                    flowMissing = true;
                    break;
                }
            }
        }
        if(flowMissing) {
            flow_util::computeMissingFlow(file_io);
        }

        DynamicConfidence dynamicConfidence(file_io, 4.0);

        vector<Vec3b> validColor{Vec3b(128,0,0), Vec3b(192,192,128), Vec3b(192,128,128)};
	    for(auto fid=startid; fid<=endid; fid+=5) {
            Depth confidence;
            dynamicConfidence.run(fid, confidence);

            //semantic mask
            sprintf(buffer, "%s/segnet/seg%05d.png", file_io.getDirectory().c_str(), fid);
            cout << "Semantic mask loaded: " << buffer << endl;
            Mat segMask = imread(buffer);
            CHECK(segMask.data);
            cv::resize(segMask, segMask, cv::Size(confidence.getWidth(), confidence.getHeight()), 0, 0, INTER_NEAREST);
            cvtColor(segMask, segMask, CV_BGR2RGB);
            for (auto y = 0; y < segMask.rows; ++y) {
                for (auto x = 0; x < segMask.cols; ++x) {
                    Vec3b pix = segMask.at<Vec3b>(y, x);
                    if(std::find(validColor.begin(), validColor.end(), pix) == validColor.end())
                        confidence(x, y) = 0.0;
                }
            }

            confidence.saveDepthFile(file_io.getDynamicConfidence(fid));
            confidence.saveImage(file_io.getDynamicConfidenceImage(fid), 5);
        }
    }catch(const std::exception& e){
        cerr << e.what() << endl;
    }
    return 0;
}

