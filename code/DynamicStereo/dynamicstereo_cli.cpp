//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
#include "../GeometryModule/dynamicwarpping.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(tWindow, 100, "tWindow");
DEFINE_int32(stereo_stride, 2, "tWindowStereo");
DEFINE_int32(downsample, 2, "downsample ratio");
DEFINE_int32(resolution, 128, "disparity resolution");
DEFINE_int32(stereo_interval, 5, "interval for stereo");
DEFINE_double(weight_smooth, 0.25, "smoothness weight for stereo");
DEFINE_string(classifierPath, "", "not used");
DEFINE_double(min_disp, -1, "min disp");
DEFINE_double(max_disp, -1, "max_disp");
DEFINE_bool(noOptimize, false, "only compute matching const");

DECLARE_string(flagfile);

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
	if (!stlplus::folder_exists(file_io.getDirectory() + "/midres"))
		stlplus::folder_create(file_io.getDirectory() + "/midres");
	if (!stlplus::folder_exists(file_io.getDirectory() + "/midres/prewarp"))
		stlplus::folder_create(file_io.getDirectory() + "/midres/prewarp");

	Mat refImage = imread(file_io.getImage(FLAGS_testFrame));
	CHECK(refImage.data) << file_io.getImage(FLAGS_testFrame);
	const int width = refImage.cols;
	const int height = refImage.rows;

	DynamicStereo stereo(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_stereo_stride, FLAGS_downsample,
						 FLAGS_weight_smooth,
						 FLAGS_resolution, FLAGS_min_disp, FLAGS_max_disp);


	{
//			    //test SfM
		const int tf1 = FLAGS_testFrame;
		Mat imgRef = imread(file_io.getImage(tf1));
//			//In original scale
		Vector2d pt(-1, -1);
		stereo.dbtx = pt[0];
		stereo.dbty = pt[1];
//			//Vector2d pt(794, 294);
//			//Vector2d pt(1077, 257);
//				sprintf(buffer, "%s/temp/epipolar%05d_ref.jpg", file_io.getDirectory().c_str(), tf1);
//				cv::circle(imgRef, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0, 0, 255), 2);
//				imwrite(buffer, imgRef);
//				for (auto tf2 = stereo.getOffset(); tf2 < stereo.getOffset() + stereo.gettWindow(); ++tf2) {
//					Mat imgL, imgR;
//					utility::verifyEpipolarGeometry(file_io, stereo.getSfMModel(), tf1, tf2, pt, imgL, imgR);
//					sprintf(buffer, "%s/temp/epipolar%05dto%05d.jpg", file_io.getDirectory().c_str(), tf1, tf2);
//					imwrite(buffer, imgR);
	}

	Depth depth;
	Mat depthMask;
	sprintf(buffer, "%s/midres/depth%05d.depth", file_io.getDirectory().c_str(), FLAGS_testFrame);
	if((!depth.readDepthFromFile(string(buffer)))) {
		printf("Running stereo for frame %d\n", FLAGS_testFrame);
		stereo.runStereo(depth, depthMask);
		depth.saveDepthFile(string(buffer));
	}else{
		Depth tempdepth;
		Mat tempMask;
		stereo.runStereo(tempdepth, tempMask, true);
	}

	shared_ptr<DynamicWarpping> warpping(new DynamicWarpping(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_resolution));
	const int warpping_offset = warpping->getOffset();

	vector<Mat> prewarp;
	warpping->preWarping(prewarp);

	for(auto i=0; i<prewarp.size(); ++i){
		sprintf(buffer, "%s/midres/prewarp/prewarpb%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, i);
		imwrite(buffer, prewarp[i]);
	}

	return 0;
}
