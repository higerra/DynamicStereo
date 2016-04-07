//
// Created by yanhang on 4/7/16.
//

#include "dynamic_confidence.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>

using namespace std;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 60, "tWindow");
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

    try{
        flow_util::computeMissingFlow(file_io);
        DynamicConfidence confidence(file_io);
    }catch(const std::exception& e){
        cerr << e.what() << endl;
    }
    return 0;
}

