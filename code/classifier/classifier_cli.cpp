//
// Created by yanhang on 5/19/16.
//

#include <gflags/gflags.h>
#include "classifier.h"

using namespace std;
using namespace dynamic_stereo;
using namespace cv;

DEFINE_int32(downsample, 4, "downsample");
DEFINE_string(mode, "test", "mode: classifier or test");
DEFINE_int32(width, -1, "output width");
DEFINE_int32(height, -1, "output height");

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_mode == "train"){
        if(argc < 3){
            cerr << "Usage: ./classifier <path-to-trainset> <path-to-output-model>" << endl;
            return 1;
        }
        printf("Training...\n");
        float start_t = (float)cv::getTickCount();
        trainSVM(string(argv[1]), string(argv[2]));
        printf("Done. Time usage:%.2fs\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());
    }else if(FLAGS_mode == "test"){
        if(argc < 4){
            cerr << "Usage: ./classifier <path-to-testset> <path-to-model>" << endl;
            return 1;
        }
        int width = FLAGS_width / FLAGS_downsample;
        int height = FLAGS_height / FLAGS_downsample;
        CHECK_GT(width, 0);
        CHECK_GT(height, 0);
        float start_t = (float)cv::getTickCount();
        printf("Predicting...\n");
        Mat result = predictSVM(string(argv[2]), string(argv[1]), width, height);
        printf("Done. Time usage:%.2fs\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());
        imwrite(argv[3], result);
    }else{
        cerr << "Unrecognized mode, should be either 'train' or 'test'" << endl;
        return 1;
    }

    return 0;
}

