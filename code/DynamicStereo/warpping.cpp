//
// Created by Yan Hang on 4/9/16.
//

#include "dynamicstereo.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{
	void DynamicStereo::warpToAnchor(const Depth& refDisp, const cv::Mat& mask, const int startid, const int endid, std::vector<cv::Mat>& warpped) const{
		CHECK_GE(startid, 0);
		CHECK_LT(endid, images.size());
		CHECK_GT(endid, startid);

		cout << "Warpping..." << endl;
		vector<Mat> fullimages(endid-startid+1);
		//set black pixel to (1,1,1). (0,0,0) means invalid pixel
		for(auto i=startid; i<=endid; ++i) {
			fullimages[i - startid] = imread(file_io.getImage(i + offset));
			for(auto y=0; y<fullimages[i-startid].rows; ++y){
				for(auto x=0; x<fullimages[i-startid].cols; ++x){
					if(fullimages[i-startid].at<Vec3b>(y,x) == Vec3b(0,0,0))
						fullimages[i-startid].at<Vec3b>(y,x) = Vec3b(1,1,1);
				}
			}
		}
		warpped.resize(fullimages.size());

		const int w = fullimages[0].cols;
		const int h = fullimages[0].rows;
		CHECK_EQ(mask.cols, w);
		CHECK_EQ(mask.rows, h);
		CHECK_EQ(mask.channels(), 1);

		const theia::Camera cam1 = reconstruction.View(orderedId[anchor].second)->Camera();


		for(auto i=startid; i<=endid; ++i){
			cout << i+offset << ' ' << flush;
			if(i == anchor-offset) {
				warpped[i-startid] = fullimages[i-startid].clone();
				continue;
			}else{
				warpped[i-startid] = Mat(h,w,CV_8UC3,Scalar(0,0,0));
			}
			const theia::Camera cam2 = reconstruction.View(orderedId[i+offset].second)->Camera();
			for(auto y=downsample; y<h-downsample; ++y) {
				for (auto x = downsample; x < w - downsample; ++x) {
					if (mask.at<uchar>(y, x) < 200)
						continue;
					Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x, y));
					//ray.normalize();
					double disp = refDisp.getDepthAt(Vector2d(x / downsample, y / downsample));
					Vector3d spt = cam1.GetPosition() + ray * model->dispToDepth(disp);
					Vector2d imgpt;
					cam2.ProjectPoint(spt.homogeneous(), &imgpt);
					if (imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < w - 1 && imgpt[1] < h - 1) {
						Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(fullimages[i].data, w, h, imgpt);
						warpped[i].at<Vec3b>(y, x) = Vec3b(pix2[0], pix2[1], pix2[2]);
					}
				}
			}
		}
		cout << endl;
	}
}//namespace dynamic_stereo

