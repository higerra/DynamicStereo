//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "optimization.h"
#include "local_matcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{


	DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
	                             const int tWindow_, const int tWindowStereo_, const int downsample_, const double weight_smooth_, const int dispResolution_,
	                             const double min_disp_, const double max_disp_):
			file_io(file_io_), anchor(anchor_), tWindow(tWindow_), tWindowStereo(tWindowStereo_), downsample(downsample_), dispResolution(dispResolution_),
			pR(3), dbtx(-1), dbty(-1){
		CHECK_LE(tWindowStereo, tWindow);
		cout << "Reading..." << endl;
		offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
		CHECK_GE(file_io.getTotalNum(), offset + tWindow);
		cout << "Reading reconstruction" << endl;
		sfmModel.init(file_io.getReconstruction());

		CHECK(downsample == 1 || downsample == 2 || downsample == 4 || downsample == 8) << "Invalid downsample ratio!";
		images.resize((size_t)tWindow);

		cout << "Reading images" << endl;
		const int nLevel = (int)std::log2((double)downsample) + 1;
		for(auto i=0; i<tWindow; ++i){
			vector<Mat> pyramid(nLevel);
			Mat tempMat = imread(file_io.getImage(i + offset));
			pyramid[0] = tempMat;
			for(auto k=1; k<nLevel; ++k)
				pyrDown(pyramid[k-1], pyramid[k]);
			images[i] = pyramid.back().clone();

//			const theia::Camera& cam = sfmModel.getCamera(i+offset);
//			cout << "Projection matrix for view " << i+offset << endl;
//			theia::Matrix3x4d pm;
//			cam.GetProjectionMatrix(&pm);
//			cout << pm << endl;
//			double dis1 = cam.RadialDistortion1();
//			double dis2 = cam.RadialDistortion2();
//			printf("Radio distortion:(%.5ff,%.5f)\n", dis1, dis2);
		}
		CHECK_GT(images.size(), 2) << "Too few images";
		width = images.front().cols;
		height = images.front().rows;
		dispUnary.initialize(width, height, 0.0);

		model = shared_ptr<StereoModel<EnergyType> >(new StereoModel<EnergyType>(images[anchor-offset], (double)downsample, dispResolution, 1000, weight_smooth_));
	}


	void DynamicStereo::runStereo(Depth& result) {
		char buffer[1024] = {};

		initMRF();

		//read semantic mask
		sprintf(buffer, "%s/segnet/seg%05d.png", file_io.getDirectory().c_str(), anchor);
		Mat segMaskImg = imread(buffer);
		CHECK(segMaskImg.data) << buffer;
		//in ORIGINAL resolution
		cv::resize(segMaskImg, segMaskImg, cv::Size(width * downsample, height * downsample), 0,0,INTER_NEAREST);
		segMask = Mat(segMaskImg.rows, segMaskImg.cols, CV_8UC1, Scalar(255));
		sprintf(buffer, "%s/temp/seg%05d.jpg", file_io.getDirectory().c_str(), anchor);
		imwrite(buffer, segMask);

		vector<Vec3b> validColor{Vec3b(0,0,128), Vec3b(128,192,192), Vec3b(128,128,192)};
		for(auto y=0; y<segMaskImg.rows; ++y){
			for(auto x=0; x<segMaskImg.cols; ++x){
				Vec3b pix = segMaskImg.at<Vec3b>(y,x);
				if(std::find(validColor.begin(), validColor.end(), pix) == validColor.end())
					segMask.at<uchar>(y,x) = 0;
			}
		}
		sprintf(buffer, "%s/temp/segmask%05d.jpg", file_io.getDirectory().c_str(), anchor);
		imwrite(buffer, segMask);

		{
			//debug: visualize seg mask
			Mat anchorImg = imread(file_io.getImage(anchor));
			CHECK_EQ(segMaskImg.size(), anchorImg.size());
			Mat overlayImg;
			cv::addWeighted(anchorImg, 0.5, segMaskImg, 0.5, 0.0, overlayImg);
			sprintf(buffer, "%s/temp/seg_overlay%05d.jpg", file_io.getDirectory().c_str(), anchor);
			imwrite(buffer, overlayImg);
		}


		if(dbtx >= 0 && dbty >= 0){
			//debug: inspect unary term
			int dtx = (int)dbtx / downsample;
			int dty = (int)dbty / downsample;

			printf("Unary term for (%d,%d)\n", (int)dbtx, (int)dbty);
			for (auto d = 0; d < dispResolution; ++d) {
				cout << model->operator()(dty * width + dtx, d) << ' ';
			}
			cout << endl;
			printf("noisyDisp(%d,%d): %.2f\n", (int)dbtx, (int)dbty, dispUnary[dty*width+dtx]);

			//const theia::Camera &cam = reconstruction.View(orderedId[anchor].second)->Camera();
			const theia::Camera &cam = sfmModel.getCamera(anchor);
			Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(dbtx, dbty));
			//ray.normalize();

			int tdisp = (int) dispUnary(dtx, dty);
//			int tdisp = 223;
			double td = model->dispToDepth(tdisp);
			printf("Cost at d=%d: %d\n", tdisp, model->operator()(dty * width + dtx, tdisp));

			Vector3d spt = cam.GetPosition() + ray * td;
			for (auto v = 0; v < images.size(); ++v) {
				Mat curimg = imread(file_io.getImage(v + offset));
				Vector2d imgpt;
				double curdepth = sfmModel.getCamera(v+offset).ProjectPoint(
						Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
				if (imgpt[0] >= 0 || imgpt[1] >= 0 || imgpt[0] < width || imgpt[1] < height)
					cv::circle(curimg, cv::Point(imgpt[0], imgpt[1]), 1, cv::Scalar(255, 0, 0), 2);
				sprintf(buffer, "%s/temp/project_b%05d_v%05d.jpg", file_io.getDirectory().c_str(), anchor,
						v + offset);
				imwrite(buffer, curimg);
			}
		}


		cout << "Solving with first order smoothness..." << endl;
		FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), model);
		Depth result_firstOrder;
		optimizer_firstorder.optimize(result_firstOrder, 5);

		printf("Saving depth to point cloud...\n");
		disparityToDepth(result_firstOrder, result);
		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d.ply", file_io.getDirectory().c_str(), anchor);
		utility::saveDepthAsPly(string(buffer), result, images[anchor-offset], sfmModel.getCamera(anchor), downsample);

//		Depth disp_firstOrder_filtered, depth_firstOrder_filtered;
//		printf("Applying bilateral filter to depth:\n");
//		bilateralFilter(result_firstOrder, images[anchor-offset], disp_firstOrder_filtered, 11, 5, 10, 3);
//		disparityToDepth(disp_firstOrder_filtered, depth_firstOrder_filtered);
//		depth_firstOrder_filtered.updateStatics();
//		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d_filtered.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_firstOrder_filtered, images[anchor-offset], sfmModel.getCamera(anchor), downsample);
	}



	void DynamicStereo::disparityToDepth(const Depth& disp, Depth& depth){
		depth.initialize(disp.getWidth(), disp.getHeight(), -1);
		for(auto i=0; i<disp.getWidth() * disp.getHeight(); ++i) {
			if(disp[i] < 0) {
				depth[i] = -1;
				continue;
			}
			depth[i] = model->dispToDepth(disp[i]);
		}
	}

	void DynamicStereo::bilateralFilter(const Depth &input, const cv::Mat &inputImg, Depth &output,
						 const int size, const double sigmas, const double sigmar, const double sigmau) {
		CHECK_EQ(input.getWidth(), inputImg.cols);
		CHECK_EQ(input.getHeight(), inputImg.rows);
		CHECK_EQ(size % 2, 1);
		CHECK_GT(size, 2);
		CHECK_EQ(inputImg.type(), CV_8UC3) << "Guided image should be 3 channel uchar type.";
		const int R = (size - 1) / 2;
		const int width = input.getWidth();
		const int height = input.getHeight();
		output.initialize(width, height, -1);
		const uchar *pImg = inputImg.data;

		//kerner weight the computed from:
		//1. distance
		//2. color consistancy
		//3. unary confidence

		double conf_thres = 10;
		vector<double> wunary((size_t)width * height);
		for(auto i=0; i<width * height; ++i){
			vector<double> unary(dispResolution);
			for(auto j=0; j<dispResolution; ++j)
				unary[j] = model->operator()(i,j);
			auto min_pos = min_element(unary.begin(), unary.end());
			double minu = *min_pos;
			if(minu == 0){
//				printf("(%d,%d)\n", i/width, i%width);
//				for(auto j=0; j<dispResolution; ++j)
//					cout << model->operator()(i,j) << ' ' << unary[j]<<endl;
//				CHECK_GT(minu, 0);
				wunary[i] = 0.001;
				continue;
			}
			*min_pos = std::numeric_limits<double>::max();
			double seminu = *max_element(unary.begin(), unary.end());
			double conf = seminu / minu;
			if(conf > conf_thres)
				conf = conf_thres;
			wunary[i] = math_util::gaussian(conf_thres, sigmau, conf);
		}


		//apply bilateral filter
		for (auto y = 0; y < height; ++y) {
#pragma omp parallel for
			for (auto x = 0; x < width; ++x) {
				int cidx = y * width + x;
				double m = 0.0;
				double acc = 0.0;
				Vector3d pc(pImg[3 * cidx], pImg[3 * cidx + 1], pImg[3 * cidx + 2]);
				for (auto dx = -1 * R; dx <= R; ++dx) {
					for (auto dy = -1 * R; dy <= R; ++dy) {
						int curx = x + dx;
						int cury = y + dy;
						const int kid = (dy + R) * size + dx + R;
						if (curx < 0 || cury < 0 || curx >= width - 1 || cury >= height - 1)
							continue;
						int idx = cury * width + curx;
						Vector3d p2(pImg[3 * idx], pImg[3 * idx + 1], pImg[3 * idx + 2]);
						double wcolor = (p2 - pc).squaredNorm() / (sigmar * sigmar);
						double wdis = (dx * dx + dy * dy) / (sigmas * sigmas);
						double w = std::exp(-1 * (wcolor + wdis)) * wunary[idx];
						m += w;
						acc += input(curx, cury) * w;
					}
				}
				CHECK_GT(m, 0);
				output(x, y) = acc / m;
			}
		}
	}

}
