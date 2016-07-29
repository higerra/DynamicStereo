//
// Created by yanhang on 7/27/16.
//
#include <fstream>
#include "videosegmentation.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_double(c, 2.0, "parameter c");

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./VideoSegmentation <path-to-video>" << endl;
        return 1;
    }
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

    char buffer[128] = {};

    string strArg(argv[1]);
    string subfix = strArg.substr(strArg.find_last_of("."));
    string dir;
    if(strArg.find_last_of("/") == strArg.npos)
        strArg = "./" + strArg;
    cout << "strArg:" << strArg << endl;
    dir = strArg.substr(0, strArg.find_last_of("/"));
    dir.append("/");

    string out_path(dir);
    if(argc >= 3)
        out_path = string(argv[2]);
    CHECK(!subfix.empty());

    vector<string> filenames;
    if(subfix == ".mp4")
        filenames.emplace_back(strArg.substr(strArg.find_last_of("/")+1));
    else{
        ifstream listIn(argv[1]);
        CHECK(listIn.is_open()) << "Can not open list: " << argv[1];
        string temp;
        while(listIn >> temp)
            filenames.push_back(temp);
    }

    for(const auto& filename: filenames) {
        //load video
        printf("Processing video %s\n", filename.c_str());
        VideoCapture cap(dir + filename);
        CHECK(cap.isOpened()) << "Can not load video: " << argv[1];
        vector<Mat> images;
        while (true) {
            Mat frame;
            if (!cap.read(frame))
                break;
            images.push_back(frame);
        }
        //test for video segmentation based on binary descriptor
        Mat segment;
        printf("Video segmentation...\n");
        segment_video(images, segment, 9, (float) FLAGS_c, 100, 100);
        Mat segment_vis = visualizeSegmentation(segment);
        Mat result;
        cv::addWeighted(images[0], 0.1, segment_vis, 0.9, 0.0, result);

        sprintf(buffer, "%s/%s_result_c%.1f.png", out_path.c_str(), filename.substr(0, filename.find_last_of(".")).c_str(), FLAGS_c);
        printf("Writing %s\n", buffer);
        imwrite(buffer, result);
    }
    return 0;
}

