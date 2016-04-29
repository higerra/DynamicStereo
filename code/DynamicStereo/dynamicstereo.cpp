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


	void DynamicStereo::runStereo(const cv::Mat& inputMask, Depth& depth_firstOrder, cv::Mat& depthMask, bool dryrun) {
		char buffer[1024] = {};
		initMRF();
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				EnergyType min_energy = numeric_limits<EnergyType>::max();
				for (int d = 0; d < dispResolution; ++d) {
					const EnergyType curEnergy = model->operator()(y*width+x, d);
					if ((double) curEnergy < min_energy) {
						dispUnary.setDepthAtInt(x, y, (double) d);
						min_energy = curEnergy;
					}
				}
			}
		}

		cv::Mat stereoMask;
		cv::resize(inputMask, stereoMask, cv::Size(width, height), 0, 0, INTER_NEAREST);
		depthMask = Mat(height, width, CV_8UC1, Scalar(255));

		if(dbtx >= 0 && dbty >= 0){
			//debug: inspect unary term
			int dtx = (int)dbtx / downsample;
			int dty = (int)dbty / downsample;

			//const theia::Camera &cam = reconstruction.View(orderedId[anchor].second)->Camera();
			const theia::Camera &cam = sfmModel.getCamera(anchor);
			Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(dbtx, dbty));
			//ray.normalize();

			int tdisp = (int) dispUnary(dtx, dty);
//			int tdisp = 142;
			double td = model->dispToDepth(tdisp);
			for(auto d=0; d<dispResolution; ++d)
				cout << (int)model->operator()(dty*width + dtx, d) << ' ';
			cout<< endl;
			cout << "Cost at d=" << tdisp << ": " << (int)model->operator()(dty * width + dtx, tdisp) << endl;

			Vector3d spt = cam.GetPosition() + ray * td;
			for (auto v = 0; v < images.size(); ++v) {
				Mat curimg = imread(file_io.getImage(v + offset));
				Vector2d imgpt;
				double curdepth = sfmModel.getCamera(v+offset).ProjectPoint(
						Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
				if (imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < curimg.cols && imgpt[1] < curimg.rows)
					cv::circle(curimg, cv::Point(imgpt[0], imgpt[1]), 1, cv::Scalar(255, 0, 0), 2);
				sprintf(buffer, "%s/temp/project_b%05d_v%05d.jpg", file_io.getDirectory().c_str(), anchor,
						v + offset);

				imgpt = imgpt / (double)downsample;
				if(imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < images[0].cols-1 && imgpt[1] < images[0].rows-1){
					Vector3d pix = interpolation_util::bilinear<uchar,3>(images[v].data, images[v].cols, images[v].rows,  imgpt);
					//double gv = 0.114 * pix[0] + 0.587 * pix[1] + 0.299 * pix[2];
					printf("%.2f %.2f %.2f\n", pix[0], pix[1], pix[2]);
				}
				imwrite(buffer, curimg);
			}
//			cout << "===============================" << endl;
		}


//		if(dbtx >= 0 && dbty >= 0){
//			//debug for frequency confidence
//			for(int tdisp = 0; tdisp < dispResolution; ++tdisp) {
//				const double ratio = getFrequencyConfidence(anchor - offset, (int) dbtx / downsample, (int) dbty / downsample, tdisp);
//				double alpha = 3, beta=2;
//				double conf = 1 / (1 + std::exp(-1*alpha*(ratio - beta)));
//				printf("frequency confidence for (%d,%d) at disp %d: %.3f\n", (int) dbtx, (int) dbty, tdisp, conf);
//			}
//		}

		if(dryrun)
			return;

		cout << "Solving with first order smoothness..." << endl;
		FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), model);
		Depth result_firstOrder;
		optimizer_firstorder.optimize(result_firstOrder, 100);

		for(auto y=0; y<height; ++y) {
			for (auto x = 0; x < width; ++x) {
				if (stereoMask.at<uchar>(y, x) < 200) {
					result_firstOrder(x, y) = 0;
					continue;
				}
			}
		}

//		Depth depth_firstOrder;
		printf("Saving depth to point cloud...\n");
		disparityToDepth(result_firstOrder, depth_firstOrder);


		//masking out invalid region
		//remove pixel where half disparity project outof half frames
		const theia::Camera& refCam = sfmModel.getCamera(anchor);
		const double invisThreshold = 0.3;

		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x*downsample, y*downsample));
				Vector3d spt = refCam.GetPosition() + depth_firstOrder(x,y) * ray;
				double invisCount = 0.0;
				for(auto v=0; v < images.size(); ++v){
					Vector2d imgpt;
					sfmModel.getCamera(v+offset).ProjectPoint(spt.homogeneous(), &imgpt);
					if(imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] >= width * downsample || imgpt[1] >= height * downsample)
						invisCount += 1.0;
				}
				if(invisCount / (double)images.size()> invisThreshold)
					depthMask.at<uchar>(y,x) = (uchar)0;
			}
		}



		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d.ply", file_io.getDirectory().c_str(), anchor);
		utility::saveDepthAsPly(string(buffer), depth_firstOrder, images[anchor-offset], sfmModel.getCamera(anchor), downsample);

//		Depth disp_firstOrder_filtered, depth_firstOrder_filtered;
//		Depth disp_firstOrder_filtered;
//		printf("Applying bilateral filter to depth:\n");
//		bilateralFilter(result_firstOrder, images[anchor-offset], disp_firstOrder_filtered, 11, 5, 10, 3);
//		disparityToDepth(disp_firstOrder_filtered, depth_firstOrder_filtered);
//		depth_firstOrder_filtered.updateStatics();
//		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d_filtered.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_firstOrder_filtered, images[anchor-offset], sfmModel.getCamera(anchor), downsample);

		if(dbtx >=0 && dbty >= 0){
			printf("Result disparity for (%d,%d): %d\n", (int)dbtx, (int)dbty, (int)result_firstOrder((int)dbtx/downsample, (int)dbty/downsample));
		}
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

		const double max_disp_diff = 10;
		//apply bilateral filter
		for (auto y = 0; y < height; ++y) {
#pragma omp parallel for
			for (auto x = 0; x < width; ++x) {
				int cidx = y * width + x;
				double m = 0.0;
				double acc = 0.0;
				Vector3d pc(pImg[3 * cidx], pImg[3 * cidx + 1], pImg[3 * cidx + 2]);
				double disp1 = input(x,y);
				for (auto dx = -1 * R; dx <= R; ++dx) {
					for (auto dy = -1 * R; dy <= R; ++dy) {
						int curx = x + dx;
						int cury = y + dy;
						const int kid = (dy + R) * size + dx + R;
						if (curx < 0 || cury < 0 || curx >= width - 1 || cury >= height - 1)
							continue;
						double disp2 = input(curx,cury);
						if(abs(disp1 - disp2) > max_disp_diff)
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
				if(m != 0)
					output(x, y) = acc / m;
				else
					output(x,y) = input(x,y);
			}
		}
	}

}
