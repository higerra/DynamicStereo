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
DEFINE_bool(run_pixel, false, "run pixel pixel level segmentation");
DEFINE_bool(run_region, true, "run region level segmentation");
DEFINE_string(input_segment, "", "input segmentation");
DEFINE_int32(write_level, -1, "write level");
DEFINE_double(wa, 0.1, "weight for appearance");

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./VideoSegmentation <path-to-video>" << endl;
        return 1;
    }
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

    char buffer[128] = {};

    {
        Mat tmp = imread(
                "/home/yanhang/Documents/research/DynamicStereo/data/working/data_vegas22/images/image00120.jpg");
        Mat graph_seg;
        vector<vector<int> > segs;
        segment_gb::segment_image(tmp, graph_seg, segs, 3, 800, 50);
        FileStorage graph_seg_file("graph_seg_vegas22_00120.yml", FileStorage::WRITE);
        graph_seg_file << "level0" << graph_seg;

        Mat graph_seg_vis = segment_gb::visualizeSegmentation(graph_seg);
        Mat graph_seg_out;
        cv::addWeighted(tmp, 0.2, graph_seg_vis, 0.8, 0.0, graph_seg_out);
        imwrite("graph_seg_vegas22_00120.png", graph_seg_out);
    }


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

    if(!FLAGS_input_segment.empty()){
        vector<Mat> segments;
        video_segment::LoadHierarchicalSegmentation(FLAGS_input_segment, segments);
        FileStorage seg_out(argv[2], FileStorage::WRITE);
        CHECK(seg_out.isOpened());
        int write_id = (int)((float)segments.size() * 0.7);
        if(FLAGS_write_level > 0){
            write_id = FLAGS_write_level;
        }
        printf("Writing %d level to %s\n", write_id, argv[2]);
        seg_out << "kLevel" << 1;
        seg_out << "level0" << segments[write_id];
        return 0;
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

        if(FLAGS_run_pixel){
            Mat segment;
            printf("Pixel based video segmentation...\n");
            video_segment::VideoSegmentOption option(FLAGS_c);
            option.refine = false;
            option.temporal_feature_type = video_segment::COMBINED;
            //option.temporal_feature_type = video_segment::TRANSITION_PATTERN;
            option.w_appearance = (float)FLAGS_wa;
            option.w_transition = 1.0 - (float)FLAGS_wa;
            option.threshold = FLAGS_c;
            int num_segments = video_segment::segment_video(images, segment, option);
            printf("Done, number of segments: %d\n", num_segments);
            Mat segment_vis = video_segment::visualizeSegmentation(segment);

            Mat blended;
            const double blend_weight = 0.1;
            cv::addWeighted(images[0], blend_weight, segment_vis, 1.0 - blend_weight, 0.0, blended);

            sprintf(buffer, "%s/%s_result_pixel_c%05.1f.png", out_path.c_str(),
                    filename.substr(0, filename.find_last_of(".")).c_str(), FLAGS_c);
            printf("Writing %s\n", buffer);
            imwrite(buffer, blended);
        }



        if(FLAGS_run_region){
            printf("Region based video segmentation...\n");
            video_segment::VideoSegmentOption option(FLAGS_c);
            option.refine = false;
            option.temporal_feature_type = video_segment::COMBINED;
            option.pixel_feture_type = video_segment::PIXEL_VALUE;
            option.region_temporal_feature_type = video_segment::COMBINED;
            option.w_appearance = (float)FLAGS_wa;
            option.w_transition = 1.0 - (float)FLAGS_wa;

            vector<Mat> segments;
            int num_segments = video_segment::HierarchicalSegmentation(images, segments, option);
            printf("Done\n");
            sprintf(buffer, "%s/segment_%s.yml", out_path.c_str(), filename.c_str());
            video_segment::SaveHierarchicalSegmentation(string(buffer), segments);

            int write_id = (int)((float)segments.size() * 0.7);
            sprintf(buffer, "%s/segment_%s_%d.yml", out_path.c_str(), filename.c_str(), write_id);
            FileStorage level_out(string(buffer), FileStorage::WRITE);
            printf("Writing %d level\n", write_id);
            level_out << "kLevel" << 1;
            level_out << "level0" << segments[write_id];


            for(auto i=0; i<segments.size(); ++i) {
                Mat segment_vis = video_segment::visualizeSegmentation(segments[i]);
                Mat blended;
                const double blend_weight = 0.1;
                cv::addWeighted(images[images.size() / 2], blend_weight, segment_vis, 1.0 - blend_weight, 0.0, blended);
                sprintf(buffer, "%s/%s_result_region_l%05d.png", out_path.c_str(),
                        filename.substr(0, filename.find_last_of(".")).c_str(), i);
                printf("Writing %s\n", buffer);
                imwrite(buffer, blended);
            }
        }
    }
    return 0;
}

