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
DEFINE_int32(tWindow, 60, "tWindow");
DEFINE_int32(tWindowStereo, 30, "tWindowStereo");
DEFINE_int32(downsample, 4, "downsample ratio");
DEFINE_int32(resolution, 256, "disparity resolution");
DEFINE_double(weight_smooth, 0.001, "smoothness weight for stereo");
DEFINE_double(min_disp, -1, "minimum disparity");
DEFINE_double(max_disp, -1, "maximum disparity");

int main(int argc, char **argv) {
	if (argc < 2) {
		cerr << "Usage: DynamicStereo <path-to-data>" << endl;
		return 1;
	}

	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

	FileIO file_io(argv[1]);
	CHECK_GT(file_io.getTotalNum(), 0);
	char buffer[1024] = {};

	if (!stlplus::folder_exists(file_io.getDirectory() + "/temp"))
		stlplus::folder_create(file_io.getDirectory() + "/temp");

	DynamicStereo stereo(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_tWindowStereo, FLAGS_downsample, FLAGS_weight_smooth,
	                     FLAGS_resolution, FLAGS_min_disp, FLAGS_max_disp);

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

	{
		//test graph based segmentation
//		cout << "Test graph based segmentation" << endl;
//		Mat sgtest = imread(file_io.getImage(FLAGS_testFrame));
//		float spTest[] = {1, 1.5, 10};
//		float mults[] = {3,5,8,12,24,50,100};
//		for(auto cid=0; cid < 7; ++cid){
//			cout << "Configuration: " << cid << endl << flush;
//			Mat output;
//			vector<vector<int> > seg;
//			//int nLabel = segment_gb::segment_image(sgtest, output, seg, spTest[0] * mults[cid], spTest[1] * mults[cid], (int)spTest[1] * (int)mults[cid]);
//			int nLabel = segment_gb::segment_image(sgtest, output, seg, 0.8, 500, 30);
//			sprintf(buffer, "%s/temp/gbseg%05d_%03d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, cid);
//			imwrite(buffer, output);
//		}
	}


    {
        //    //test SfM
        const int tf1 = FLAGS_testFrame;
		Mat imgRef = imread(file_io.getImage(tf1));
        //In original scale
        Vector2d pt(1078,258);
		sprintf(buffer, "%s/temp/epipolar%05d_ref.jpg", file_io.getDirectory().c_str(), tf1);
		cv::circle(imgRef, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0,0,255),2);
		imwrite(buffer, imgRef);
        for (auto tf2 = stereo.getOffset(); tf2 < stereo.getOffset() + stereo.gettWindow(); ++tf2) {
			Mat imgL, imgR;
            stereo.verifyEpipolarGeometry(tf1, tf2, pt, imgL, imgR);
            sprintf(buffer, "%s/temp/epipolar%05dto%05d.jpg", file_io.getDirectory().c_str(), tf1, tf2);
            imwrite(buffer, imgR);
        }
    }

    stereo.runStereo();
    //stereo.warpToAnchor();


    return 0;
}
