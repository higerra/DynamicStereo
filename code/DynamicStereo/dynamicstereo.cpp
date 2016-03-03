//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "optimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{

	namespace segment_util{
		void visualizeSegmentation(const std::vector<int>& labels, const int width, const int height, cv::Mat& output){
			CHECK_EQ(width * height, labels.size());
			output = Mat(height, width, CV_8UC3);

			std::vector<cv::Vec3b> colorTable((size_t)width * height);
			std::default_random_engine generator;
			std::uniform_int_distribution<int> distribution(0, 255);

			for (int i = 0; i < width * height; i++) {
				for(int j=0; j<3; ++j)
					colorTable[i][j] = (uchar)distribution(generator);
			}

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++)
					output.at<cv::Vec3b>(y,x) = colorTable[labels[y*width + x]];
			}
		}
	}

	DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
	                             const int tWindow_, const int downsample_, const double weight_smooth_, const int dispResolution_,
	                             const double min_disp_, const double max_disp_):
			file_io(file_io_), anchor(anchor_), tWindow(tWindow_), downsample(downsample_), dispResolution(dispResolution_), pR(3),
			weight_smooth(weight_smooth_), MRFRatio(10000),
			min_disp(min_disp_), max_disp(max_disp_), dispScale(1000){

		cout << "Reading..." << endl;
		offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
		CHECK_GE(file_io.getTotalNum(), offset + tWindow);
		cout << "Reading reconstruction" << endl;
		CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Can not open reconstruction file";
		CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());
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

			const theia::Camera cam = reconstruction.View(i+offset)->Camera();
			cout << "Projection matrix for view " << i+offset<< endl;
			theia::Matrix3x4d pm;
			cam.GetProjectionMatrix(&pm);
			cout << pm << endl;
		}
		CHECK_GT(images.size(), 2) << "Too few images";
		width = images.front().cols;
		height = images.front().rows;

		dispUnary.initialize(width, height, 0.0);

		cout << "Computing disparity range" << endl;
		if(min_disp < 0 || max_disp < 0)
			computeMinMaxDisparity();
	}

	void DynamicStereo::verifyEpipolarGeometry(const int id1, const int id2,
	                                           const Eigen::Vector2d& pt,
	                                           cv::Mat &imgL, cv::Mat &imgR) {
		CHECK_GE(id1 - offset, 0);
		CHECK_GE(id2 - offset, 0);
		CHECK_LT(id1 - offset, images.size());
		CHECK_LT(id2 - offset, images.size());
		CHECK_GE(pt[0], 0);
		CHECK_GE(pt[1], 0);
		CHECK_LT(pt[0], (double)width * downsample);
		CHECK_LT(pt[1], (double)height * downsample);

		theia::Camera cam1 = reconstruction.View(id1)->Camera();
		theia::Camera cam2 = reconstruction.View(id2)->Camera();

		Vector3d ray1 = cam1.PixelToUnitDepthRay(pt*downsample);
		ray1.normalize();

		imgL = images[id1-offset].clone();
		imgR = images[id2-offset].clone();

		cv::circle(imgL, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0,0,255), 2);

		const double min_depth = 1.0 / max_disp;
		const double max_depth = 1.0 / min_disp;
		printf("min depth:%.3f, max depth:%.3f\n", min_depth, max_depth);

		double cindex = 0.0;
		double steps = 1000;
		for(double i=min_disp; i<max_disp; i+=(max_disp-min_disp)/steps){
			Vector3d curpt = cam1.GetPosition() + ray1 * 1.0 / i;
			Vector4d curpt_homo(curpt[0], curpt[1], curpt[2], 1.0);
			Vector2d imgpt;
			double depth = cam2.ProjectPoint(curpt_homo, &imgpt);
			if(depth < 0)
				continue;
			imgpt = imgpt / (double)downsample;
			//printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
			cv::Point cvpt(((int)imgpt[0]), ((int)imgpt[1]));
			cv::circle(imgR, cvpt, 1, cv::Scalar(255 - cindex * 255.0, 0 ,cindex * 255.0));
			cindex += 1.0 / steps;
		}
	}



	void DynamicStereo::runStereo() {
		char buffer[1024] = {};
		//debug for sample patch
//        vector<vector<double> > testP(2);
//        const int tf2 = 3;
//        Vector2d tloc1 = Vector2d(680,387) / downsample;
//        Vector2d tloc2 = Vector2d(100,100) / downsample;
//        MRF_util::samplePatch(images[anchor-offset], tloc1, pR, testP[0]);
//        MRF_util::samplePatch(images[tf2-offset], tloc2, pR, testP[1]);
//        double testncc = MRF_util::medianMatchingCost(testP, 0);
//        cout << "Test ncc: " << testncc << endl;

		initMRF();

		{
			//debug: inspect unary term
			const int tx = 13;
			const int ty = 30;
			printf("Unary term for (%d,%d)\n", tx, ty);
			for (auto d = 0; d < dispResolution; ++d) {
				cout << MRF_data[dispResolution * (ty * width + tx) + d] << ' ';
			}
			cout << endl;
			printf("noisyDisp(%d,%d): %.2f\n", tx, ty, dispUnary.getDepthAtInt(tx, ty));
		}

		//generate proposal
		sprintf(buffer, "%s/temp/unarydisp_b%05d.jpg", file_io.getDirectory().c_str(), anchor);
		dispUnary.saveImage(string(buffer), 255.0 / (double)dispResolution);

//		cout << "Generating plane proposal" << endl;
//		ProposalSegPlnMeanshift proposalFactoryMeanshift(file_io, images[anchor-offset], dispUnary, dispResolution, min_disp, max_disp);
//		vector<Depth> proposals;
//		proposalFactoryMeanshift.genProposal(proposals);
//
////		vector<Depth> proposalsGb;
////		ProposalSegPlnGbSegment proposalFactoryGbSegment(file_io, images[anchor-offset], dispUnary, dispResolution, min_disp, max_disp);
////		proposalFactoryGbSegment.genProposal(proposalsGb);
////		proposals.insert(proposals.end(), proposalsGb.begin(), proposalsGb.end());
//		for(auto i=0; i<proposals.size(); ++i){
//			sprintf(buffer, "%s/temp/proposalPln%05d_%03d.jpg", file_io.getDirectory().c_str(), anchor, i);
//			proposals[i].saveImage(buffer, 255.0 / (double)dispResolution);
//		}

		//fusion move
//		cout << "Solving with second order smoothness..." << endl;
//		Depth currentBest = dispUnary;
//		for(auto i=0; i<proposals.size(); ++i){
//			cout << "=======================" << endl;
//			cout << "Fusing current best and proposal " << i << endl;
//			fusionMove(currentBest, proposals[i]);
//		}
//		sprintf(buffer, "%s/temp/secondOrder%05d_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		currentBest.saveImage(string(buffer), 255.0 / (double)dispResolution);
//
//
		cout << "Solving with first order smoothness..." << endl;
		FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), images[anchor-offset], MRF_data, (float)MRFRatio, dispResolution, (EnergyType)(MRFRatio * weight_smooth));
		Depth result_firstOrder;
		optimizer_firstorder.optimize(result_firstOrder, 10);
		sprintf(buffer, "%s/temp/firstOrder%05d_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
		result_firstOrder.saveImage(buffer, 255.0 / (double)dispResolution);
	}

	void DynamicStereo::warpToAnchor() const{
//		cout << "Warpping..." << endl;
//		vector<Mat> fullimages(images.size());
//		for(auto i=0; i<fullimages.size(); ++i)
//			fullimages[i] = imread(file_io.getImage(i+offset));
//		vector<Mat> warpped(fullimages.size());
//
//		const int w = fullimages[0].cols;
//		const int h = fullimages[0].rows;
//
//		const theia::Camera cam1 = reconstruction.View(anchor)->Camera();
//
//		for(auto i=0; i<fullimages.size(); ++i){
//			cout << i+offset << ' ' << flush;
//			warpped[i] = fullimages[anchor-offset].clone();
//			if(i == anchor-offset)
//				continue;
//			const theia::Camera cam2 = reconstruction.View(i+offset)->Camera();
//			for(auto y=downsample; y<h-downsample; ++y){
//				for(auto x=downsample; x<w-downsample; ++x){
//					Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x,y));
//					ray.normalize();
//					double depth = refDisparity.getDepthAt(Vector2d(x/downsample, y/downsample));
//					Vector3d spt = cam1.GetPosition() + ray * depth;
//					Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
//					Vector2d imgpt;
//					cam2.ProjectPoint(spt_homo, &imgpt);
//					if(imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < w -1 && imgpt[1] < h - 1){
//						Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(fullimages[i].data, w, h, imgpt);
//						warpped[i].at<Vec3b>(y,x) = Vec3b(pix2[0], pix2[1], pix2[2]);
//					}
//				}
//			}
//		}
//
//		cout << endl << "Saving..." << endl;
//		char buffer[1024] = {};
//		for(auto i=0; i<warpped.size(); ++i){
//			sprintf(buffer, "%s/temp/warpped_b%05d_f%05d.jpg", file_io.getDirectory().c_str(),  anchor, i+offset);
//			imwrite(buffer, warpped[i]);
//		}
	}
}
