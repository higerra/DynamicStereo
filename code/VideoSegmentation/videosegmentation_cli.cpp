//
// Created by yanhang on 7/27/16.
//
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "videosegmentation.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_double(c, 20.0, "parameter c");
DEFINE_double(theta, 100, "parameter theta");

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
    if(subfix != ".txt")
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
        printf("=================================\n");
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
        Mat segment;//, segment_refine;
        printf("Video segmentation...\n");
        video_segment::VideoSegmentOption option(FLAGS_c);
        option.refine = false;
        option.temporal_feature_type = video_segment::COMBINED;
        //option.temporal_feature_type = video_segment::TRANSITION_PATTERN;
        int num_segments = video_segment::segment_video(images, segment, option);
        printf("Done, number of segments: %d\n", num_segments);

        Mat segment_vis = video_segment::visualizeSegmentation(segment);

        Mat result, result_refined;
        const double blend_weight = 0.2;
        cv::addWeighted(images[0], blend_weight, segment_vis, 1.0 - blend_weight, 0.0, result);


        sprintf(buffer, "%s/%s_result_c%05.1f.png", out_path.c_str(), filename.substr(0, filename.find_last_of(".")).c_str(), FLAGS_c);
        printf("Writing %s\n", buffer);
        imwrite(buffer, result);
    }
    return 0;
}

