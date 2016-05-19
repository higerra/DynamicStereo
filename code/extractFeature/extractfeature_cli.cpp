//
// Created by yanhang on 5/15/16.
//
#include "extracfeature.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_int32(downsample, 4, "downsample ratio");
DEFINE_int32(tWindow, 100, "tWindow");
DEFINE_bool(trainData, true, "training data");
DEFINE_int32(kBin, 8, "kBin");
DEFINE_double(min_diff, 10, "min_diff");

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: extractFeature <path-to-data>" << endl;
        return 1;
    }
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    CHECK_EQ(argc, 3);

    DataSet data_all;

    if (FLAGS_trainData) {
        string list_path = string(argv[1]);
        size_t dashPos;
        for(dashPos = list_path.size()-1; dashPos>=0; --dashPos){
            if(list_path[dashPos] == '/')
                break;
        }
        string directory = list_path.substr(0, dashPos);
        directory.append("/");
        ifstream fin(list_path.c_str());
        //cout << "list path:" << list_path << endl;
        CHECK(fin.is_open());
        while (true) {
            DataSet curData;
            string video_path, gt_path;
            fin >> video_path >> gt_path;
            cout << "video path: " << video_path << " mask path: " << gt_path << endl;
            if (video_path.empty() || gt_path.empty())
                break;
            vector<vector<float> > data;
            printf("Reading...\n");
            cv::Size dim = Feature::importData(directory + video_path, data, FLAGS_downsample, FLAGS_tWindow);
            Mat gt = imread(directory + gt_path, false);
            CHECK(gt.data);
            cv::resize(gt,gt,dim,0,0,INTER_NEAREST);
            printf("Extracting...\n");
            Feature::extractFeature(data, dim, gt, curData, FLAGS_kBin, (float)FLAGS_min_diff, Feature::RGB_CAT);
            data_all.appendDataSet(curData);
        }
    } else {
        vector<vector<float> > data;
        printf("Preparing testing data: %s\n", argv[1]);
        cv::Size dim = Feature::importData(string(argv[1]), data, FLAGS_downsample, FLAGS_tWindow);
        printf("Extracting...\n");
        Feature::extractFeature(data, dim, Mat(), data_all, FLAGS_kBin, (float)FLAGS_min_diff, Feature::RGB_CAT);
    }

    printf("Saving...\n");
    data_all.dumpData_libsvm(string(argv[2]));
	printf("Done\n");
	data_all.printStat();

    return 0;
}
