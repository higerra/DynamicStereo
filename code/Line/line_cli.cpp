//
// Created by yanhang on 3/24/16.
//

#include <iostream>
#include "lineSeg.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: Line <path-to-data>" << endl;
        return 0;
    }

    FileIO file_io(argv[1]);
    CHECK_LT(file_io.getTotalNum(), 0) << "Empty dataset";
    return 0;
}

