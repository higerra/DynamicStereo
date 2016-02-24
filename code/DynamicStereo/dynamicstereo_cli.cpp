//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 72, "tWindow");
DEFINE_int32(downsample, 2, "downsample ratio");

int main(int argc, char **argv){
    if(argc < 2){
        cerr << "Usage: DynamicStereo <path-to-data>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    FileIO file_io(argv[1]);
    CHECK_GT(file_io.getTotalNum(), 0);
    char buffer[1024] = {};

    if(!stlplus::folder_exists(file_io.getDirectory()+"temp"))
        stlplus::folder_create(file_io.getDirectory()+"temp");

    DynamicStereo stereo(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_downsample);

    //test SfM
    Mat imgL, imgR;
    const int tf1 = 60;
    const Vector2d pt(958, 444);
    for(auto tf2 = stereo.getOffset(); tf2 < stereo.getOffset() + stereo.gettWindow(); ++tf2) {
        stereo.verifyEpipolarGeometry(tf1, tf2, pt, imgL, imgR);
        CHECK_EQ(imgL.size(), imgR.size());
        Mat imgAll;
        cv::hconcat(imgL, imgR, imgAll);
        sprintf(buffer, "%s/temp/epipolar%05dto%05d.jpg", file_io.getDirectory().c_str(), tf1, tf2);
        imwrite(buffer, imgAll);
    }
    return 0;
}
