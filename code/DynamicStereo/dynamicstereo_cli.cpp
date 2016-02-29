//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 72, "tWindow");
DEFINE_int32(downsample, 4, "downsample ratio");
DEFINE_int32(resolution, 64, "disparity resolution");
DEFINE_double(weight_smooth, 0.008, "smoothness weight for stereo");

int main(int argc, char **argv){
    if(argc < 2){
        cerr << "Usage: DynamicStereo <path-to-data>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    FileIO file_io(argv[1]);
    CHECK_GT(file_io.getTotalNum(), 0);
    char buffer[1024] = {};

    if(!stlplus::folder_exists(file_io.getDirectory()+"temp"))
        stlplus::folder_create(file_io.getDirectory()+"temp");

    DynamicStereo stereo(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_downsample, FLAGS_weight_smooth, FLAGS_resolution);

//    {
//        //test mean shift segmentation
//        cout << "Test meanshift segmentation..." << endl;
//        Mat sgtest = imread(file_io.getImage(FLAGS_testFrame));
//        segment_ms::msImageProcessor ms_segmentator;
//        ms_segmentator.DefineImage(sgtest.data, segment_ms::COLOR, sgtest.rows, sgtest.cols);
//        float spTest[] = {1, 1.5, 10};
//        Vec3b colorTable[] = {Vec3b(255, 0, 0), Vec3b(0, 255, 0), Vec3b(0, 0, 255), Vec3b(255, 255, 0),
//                              Vec3b(255, 0, 255), Vec3b(0, 255, 255)};
//        for (auto cid = 1; cid <= 7; ++cid) {
//            cout << "Configuration: " << cid << endl << flush;
//            ms_segmentator.Segment((int) spTest[0] * cid, spTest[1] * (float) cid, (int) spTest[2] * cid,
//                                   meanshift::MED_SPEEDUP);
//
//            int* labels = NULL;
//            labels = ms_segmentator.GetLabels();
//            CHECK(labels);
//            const int w = sgtest.cols;
//            const int h = sgtest.rows;
//            Mat segTestRes(h, w, sgtest.type());
//            for (auto y = 0; y < h; ++y) {
//                for (auto x = 0; x < w; ++x) {
//                    int l = labels[y * w + x];
//                    segTestRes.at<Vec3b>(y,x) = colorTable[l%6];
//                }
//            }
//            sprintf(buffer, "%s/temp/meanshift%05d_%03d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, cid);
//            imwrite(buffer, segTestRes);
//        }
//    }


    //    //test SfM
//    Mat imgL, imgR;
//    const int tf1 = FLAGS_testFrame;
//    //In original scale
//    Vector2d pt(298, 181);
//    for(auto tf2 = stereo.getOffset(); tf2 < stereo.getOffset() + stereo.gettWindow(); ++tf2) {
//        stereo.verifyEpipolarGeometry(tf1, tf2, pt/(double)stereo.getDownsample(), imgL, imgR);
//        CHECK_EQ(imgL.size(), imgR.size());
//        Mat imgAll;
//        cv::hconcat(imgL, imgR, imgAll);
//        sprintf(buffer, "%s/temp/epipolar%05dto%05d.jpg", file_io.getDirectory().c_str(), tf1, tf2);
//        imwrite(buffer, imgAll);
//    }
//
    stereo.runStereo();
    //stereo.warpToAnchor();


    return 0;
}
