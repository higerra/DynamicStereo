//
// Created by yanhang on 3/24/16.
//

#include <iostream>
#include <gflags/gflags.h>

#include "lineSeg.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 60, "tWindow");

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: Line <path-to-data>" << endl;
        return 0;
    }
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    FileIO file_io(argv[1]);
    CHECK_GT(file_io.getTotalNum(), 0) << "Empty dataset";

    LineSeg lineseg(file_io);
    lineseg.runLSD();
    return 0;
}

