//
// Created by yanhang on 11/7/16.
//

#include "auto_cinemagraph.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    if(argc < 3){
        cerr << "Usage: ./AutoCinemagraph <input> <output>" << endl;
        return 1;
    }

    vector<Mat> images;
    dynamic_stereo::LoadVideo(string(argv[1]), images);
    vector<Mat> opt_flow;
    dynamic_stereo::ComputeOpticalFlow(images, opt_flow);

    vector<Mat> scores;
    dynamic_stereo::GetPixelScore(opt_flow, scores);

    vector<vector<int> > opt_spatial;
    printf("Finding optimal spatial interval...\n");
    dynamic_stereo::OptimalSpatialInterval(scores, opt_spatial);
    Mat ref = images[images.size() / 2].clone();
    Mat mask(ref.size(), CV_8UC3, Scalar(255,0,0));
    for(const auto& spatial: opt_spatial) {
        for (auto y = spatial[1]; y < spatial[3]; ++y) {
            for (auto x = spatial[0]; x < spatial[2]; ++x) {
                mask.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }
    Mat vis;
    addWeighted(ref, 0.2, mask, 0.8,0.0, vis);
    imwrite(argv[2], vis);
    return 0;
}
