//
// Created by yanhang on 6/7/16.
//
#include "contourdev.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if(argc < 2){
        cerr << "Usage: ./Contour <path-to-data>" << endl;
        return 1;
    }
    return 0;
}


