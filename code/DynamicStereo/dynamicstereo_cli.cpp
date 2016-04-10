//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
#include "dynamicwarpping.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 60, "tWindow");
DEFINE_int32(tWindowStereo, 30, "tWindowStereo");
DEFINE_int32(downsample, 4, "downsample ratio");
DEFINE_int32(resolution, 256, "disparity resolution");
DEFINE_int32(stereo_interval, 10, "interval for stereo");
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

	vector<Depth> depths;
	vector<int> depthInd;

	//run stereo
	for(auto tf = FLAGS_testFrame -FLAGS_tWindow/2; tf<=FLAGS_testFrame+FLAGS_tWindow/2; tf+=FLAGS_stereo_interval) {
		DynamicStereo stereo(file_io, tf, FLAGS_tWindow, FLAGS_tWindowStereo, FLAGS_downsample,
							 FLAGS_weight_smooth,
							 FLAGS_resolution, FLAGS_min_disp, FLAGS_max_disp);


		{
//			    //test SfM
//			const int tf1 = FLAGS_testFrame;
//			Mat imgRef = imread(file_io.getImage(tf1));
////			//In original scale
//			Vector2d pt(543, 122);
//			stereo.dbtx = pt[0];
//			stereo.dbty = pt[1];
//			//Vector2d pt(794, 294);
//			//Vector2d pt(1077, 257);
//			sprintf(buffer, "%s/temp/epipolar%05d_ref.jpg", file_io.getDirectory().c_str(), tf1);
//			cv::circle(imgRef, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0, 0, 255), 2);
//			imwrite(buffer, imgRef);
//			for (auto tf2 = stereo.getOffset(); tf2 < stereo.getOffset() + stereo.gettWindow(); ++tf2) {
//				Mat imgL, imgR;
//				stereo.verifyEpipolarGeometry(tf1, tf2, pt, imgL, imgR);
//				sprintf(buffer, "%s/temp/epipolar%05dto%05d.jpg", file_io.getDirectory().c_str(), tf1, tf2);
//				imwrite(buffer, imgR);
//			}
		}

		Depth curdepth;
		printf("Running stereo for frame %d\n", tf);
		stereo.runStereo(curdepth);
		depths.push_back(curdepth);
		depthInd.push_back(tf);
		//stereo.warpToAnchor();
	}


	//warpping
	DynamicWarpping warpping(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_downsample, depths, depthInd);
	vector<Mat> warpped;

    return 0;
}
