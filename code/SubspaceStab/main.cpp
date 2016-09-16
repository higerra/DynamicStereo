#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "subspacestab.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace cv;

DEFINE_int32(tWindow, 40, "tWindow");
DEFINE_int32(stride, 5, "Stride");
DEFINE_int32(kernelR, -1, "Radius of kernel");
DEFINE_int32(num_thread, 6, "number of threads");
DEFINE_bool(crop, true, "crop the output video");
DEFINE_bool(draw_points, false, "draw feature points");
DEFINE_bool(resize, true, "resize to 640 * 360");

DEFINE_bool(verbose, true, "Print running information");

void importVideo(const std::string& path, std::vector<cv::Mat>& images, double& fps, int& vcodec);
int main(int argc, char** argv){
    if(argc < 2){
	cerr << "Usage ./SubspaceStab <path-to-input> <path-to-output>" << endl;
	cerr << "Use './SubspaceStab --help' to view full list of options" << endl;
	return 1;
    }
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_verbose)
	FLAGS_logtostderr = true;

    vector<Mat> images;
    LOG(INFO) << "Reading video...";
    int vcodec; double frameRate;
    importVideo(string(argv[1]), images, frameRate, vcodec);

    LOG(INFO) << "Running stabilization...";
    substab::SubSpaceStabOption option(FLAGS_tWindow, FLAGS_stride, FLAGS_kernelR,
                                       FLAGS_resize, FLAGS_crop, FLAGS_draw_points, FLAGS_num_thread);
    vector<Mat> output;
    substab::subSpaceStabilization(images, output, option);

    CHECK_EQ(images.size(), output.size());

    string outputFileName;
    if(argc > 2)
        outputFileName = argv[2];
    else{
        printf("Output file name not provided. Write to 'stabilized.mp4'\n");
        outputFileName = "stabilized.avi";

    }
    
    LOG(INFO) << "Writing output...";
    cv::Size frameSize(output[0].cols, output[0].rows);
    VideoWriter writer(outputFileName, CV_FOURCC('x','2','6','4'), frameRate, frameSize);
    CHECK(writer.isOpened()) << "Can not open video to write: " << outputFileName;
    for(auto v=0; v<output.size(); ++v){
	writer << output[v];
    }
    LOG(INFO) << "All done";
    return 0;
}

void importVideo(const std::string& path, std::vector<cv::Mat>& images, double& fps, int& vcodec){
    VideoCapture cap(path);
    CHECK(cap.isOpened()) << "Can not open video " << path;
    cv::Size dsize(640,320);
    while(true){
	Mat frame;
	bool success = cap.read(frame);
	if(!success)
	    break;
	if(FLAGS_resize)
	    cv::resize(frame, frame, dsize);
	images.push_back(frame);
    }
    fps = cap.get(CV_CAP_PROP_FPS);
    vcodec = (int)cap.get(CV_CAP_PROP_FOURCC);

}


