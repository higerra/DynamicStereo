//
// Created by yanhang on 5/15/16.
//
#include "extracfeature.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(downsample, 4, "downsample ratio");
DEFINE_int32(tWindow, -1, "tWindow");
DEFINE_string(mode, "train", "mode selection: train or test");
DEFINE_int32(kBin, 10, "kBin");
DEFINE_double(min_diff, -1, "min_diff");
DEFINE_bool(csv, true, "dump out csv file");
DEFINE_string(type, "pixel", "feature type: pixel or region");
DEFINE_string(format, "RGB", "RGB or LUV");

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: extractFeature <path-to-data>" << endl;
        return 1;
    }
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    CHECK_EQ(argc, 3);

    DataSet data_all;
    char buffer[1024] = {};

    Feature::FeatureType featType;
    if(FLAGS_format == "RGB")
        featType = Feature::RGB_HIST;
    else if(FLAGS_format == "LUV")
        featType = Feature::LUV_HIST;
    else
        CHECK(true) << "Unsupported pixel format " << FLAGS_format;

    if (FLAGS_mode == "train") {
        string list_path = string(argv[1]);
        size_t dashPos;
        for(dashPos = list_path.size()-1; dashPos>=0; --dashPos){
            if(list_path[dashPos] == '/')
                break;
        }
        string directory = list_path.substr(0, dashPos);
        directory.append("/");
        ifstream fin(list_path.c_str());
        CHECK(fin.is_open());
        while (true) {
            DataSet curData;
            string video_path, gt_path;
            fin >> video_path >> gt_path;
            cout << "video path: " << video_path << " mask path: " << gt_path << endl;
            if (video_path.empty() || gt_path.empty())
                break;

            DataSet data_test;
            if(FLAGS_type == "pixel") {
                vector<vector<float> > data;
                printf("Reading...\n");
                cv::Size dim = Feature::importData(directory + video_path, data, FLAGS_downsample, FLAGS_tWindow, false, FLAGS_format);
                Mat gt = imread(directory + gt_path, false);
                CHECK(gt.data);
                cv::resize(gt, gt, dim, 0, 0, INTER_NEAREST);
                printf("Extracting...\n");
                Feature::extractFeature(data, dim, gt, curData, FLAGS_kBin, (float) FLAGS_min_diff, featType);
                data_all.appendDataSet(curData);
                //also create a test set
                Feature::extractFeature(data, dim, Mat(), data_test, FLAGS_kBin, (float) FLAGS_min_diff,
                                        featType);
            }else if(FLAGS_type == "region"){
                vector<Mat> input;
                Feature::importDataMat(directory+video_path, input, FLAGS_downsample, FLAGS_tWindow);
                vector<vector<Vector2i> > cluster;
                Feature::clusterRGBStat(input, cluster);
                cerr << "Not fininshed..." << endl;
                return 1;
            }

            if(FLAGS_csv) {
                sprintf(buffer, "%s/testFromTrain_%s.csv", directory.c_str(), video_path.c_str());
                data_test.dumpData_csv(string(buffer));
            }else{
                sprintf(buffer, "%s/testFromTrain_%s.txt", directory.c_str(), video_path.c_str());
                data_test.dumpData_libsvm(string(buffer));
            }
        }
    } else if(FLAGS_mode == "test") {
        printf("Preparing testing data: %s\n", argv[1]);
        if(FLAGS_type == "pixel") {
            vector<vector<float> > data;
            cv::Size dim = Feature::importData(string(argv[1]), data, FLAGS_downsample, FLAGS_tWindow, true, FLAGS_format);
            printf("Extracting...\n");
            Feature::extractFeature(data, dim, Mat(), data_all, FLAGS_kBin, (float) FLAGS_min_diff, featType);
        }else if(FLAGS_type == "region"){
            vector<cv::Mat> input;
            cv::Size dim = Feature::importDataMat(string(argv[1]), input, FLAGS_downsample, FLAGS_tWindow);
            vector<vector<Vector2i> > cluster;
            Feature::clusterRGBStat(input, cluster);
            cerr << "Not fininshed..." << endl;
            return 1;
        }
    }else{
        cerr << "Unsupported mode" << endl;
        return 1;
    }

    printf("Saving...\n");
    if(FLAGS_csv){
        data_all.dumpData_csv(string(argv[2]));
    }else{
        data_all.dumpData_libsvm(string(argv[2]));
    }

	printf("Done\n");
	data_all.printStat();

    return 0;
}
