//
// Created by yanhang on 5/15/16.
//
#include "tempMeanshift.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: tempMeanshift_cli <path-to-data>" << endl;
        return 1;
    }
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    string file_path(argv[1]);

    return 0;
}
