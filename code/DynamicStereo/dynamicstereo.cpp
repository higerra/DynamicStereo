//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "optimization.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "proposal.h"
#include "local_matcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{

	typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;

	namespace utility{
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



		void saveDepthToPly(const string& path, const Depth& depth, const cv::Mat& image, const theia::Camera& cam, const int downsample){
			CHECK_EQ(depth.getWidth(), image.cols);
			CHECK_EQ(depth.getHeight(), image.rows);
			TriMesh mesh;
			mesh.request_vertex_colors();
			const int& w = depth.getWidth();
			const int& h = depth.getHeight();

			vector<TriMesh::VertexHandle> vhandle((size_t)(w * h));
			Vector3d camcenter = cam.GetPosition();
			TriMesh::VertexHandle ch = mesh.add_vertex(TriMesh::Point(camcenter[0], camcenter[1], camcenter[2]));
			mesh.set_color(ch, TriMesh::Color(255,0,0));

			int vid = 0;
			for(auto y=0; y<h; ++y){
				for(auto x=0; x<w; ++x, ++vid){
					cv::Vec3b pix = image.at<Vec3b>(y,x);
					Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(x*downsample, y*downsample));
					ray.normalize();
					Vector3d spt = cam.GetPosition() + ray * depth(x,y);
					vhandle[vid] = mesh.add_vertex(TriMesh::Point(spt[0], spt[1], spt[2]));
					mesh.set_color(vhandle[vid], TriMesh::Color(pix[2], pix[1], pix[0]));
				}
			}

			OpenMesh::IO::Options wopt;
			wopt += OpenMesh::IO::Options::VertexColor;
			CHECK(OpenMesh::IO::write_mesh(mesh, path, wopt)) << "Can not write ply file " << path;
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
//			cout << "Projection matrix for view " << i+offset<< endl;
//			theia::Matrix3x4d pm;
//			cam.GetProjectionMatrix(&pm);
//			cout << pm << endl;
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

		Vector3d ray1 = cam1.PixelToUnitDepthRay(pt*(double)downsample);
		ray1.normalize();
		imgL = images[id1-offset].clone();
		imgR = images[id2-offset].clone();
		cv::circle(imgL, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0,0,255), 2);

		const double min_depth = 1.0 / max_disp;
		const double max_depth = 1.0 / min_disp;
		printf("min depth:%.3f, max depth:%.3f\n", min_depth, max_depth);

		double cindex = 0.0;
		double steps = 1000;
		for(double i=min_disp; i<=max_disp; i+=(max_disp-min_disp)/steps){
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
		//debug for NCC
//        vector<vector<double> > testP(2);
//        const int tf2 = anchor;
//        Vector2d tloc1 = Vector2d(253,44) / downsample;
//        Vector2d tloc2 = Vector2d(1070,457) / downsample;
//        local_matcher::samplePatch(images[anchor-offset], tloc1, pR, testP[0]);
//        local_matcher::samplePatch(images[tf2-offset], tloc2, pR, testP[1]);
//        double testncc = math_util::normalizedCrossCorrelation(testP[0], testP[1]);
//        cout << "Test ncc: " << testncc << endl;

		initMRF();

		{
			//debug: inspect unary term
//			const int tx = 1091 / downsample;
//			const int ty = 231 / downsample;
//			printf("Unary term for (%d,%d)\n", tx, ty);
//			for (auto d = 0; d < dispResolution; ++d) {
//				cout << MRF_data[dispResolution * (ty * width + tx) + d] << ' ';
//			}
//			cout << endl;
//			printf("noisyDisp(%d,%d): %.2f\n", tx, ty, dispUnary.getDepthAtInt(tx, ty));
//
//			const theia::Camera &cam = reconstruction.View(anchor)->Camera();
//			Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(tx * downsample, ty * downsample));
//			ray.normalize();
//
//			int tdisp = 83;
//			double td = dispToDepth(tdisp);
//			printf("Cost at d=%d: %d\n", tdisp, MRF_data[dispResolution * (ty * width + tx) + tdisp]);
//
//			Vector3d spt = cam.GetPosition() + ray * td;
//			for (auto v = 0; v < images.size(); ++v) {
//				Mat curimg = imread(file_io.getImage(v + offset));
//				Vector2d imgpt;
//				reconstruction.View(v + offset)->Camera().ProjectPoint(Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
//				if (imgpt[0] >= 0 || imgpt[1] >= 0 || imgpt[0] < width || imgpt[1] < height)
//					cv::circle(curimg, cv::Point(imgpt[0], imgpt[1]), 2, cv::Scalar(0, 0, 255), 2);
//				sprintf(buffer, "%s/temp/project_b%05d_v%05d.jpg\n", file_io.getDirectory().c_str(), anchor,
//						v + offset);
//				imwrite(buffer, curimg);
//			}
		}

		{
			//plot the gray value
//			Vector2d testPt(1178,422);
//			const int testDisp = 3;
//
//			vector<vector<double> > gv(dispResolution); //gv[i][j]: pixel in jth view on disparity i
//			const theia::Camera cam = reconstruction.View(anchor)->Camera();
//			Vector3d ray = cam.PixelToUnitDepthRay(testPt);
//			ray.normalize();
//
//			vector<Mat> grayImg(images.size());
//			for(auto v=0; v<images.size(); ++v)
//				cvtColor(images[v], grayImg[v], CV_RGB2GRAY);
//			for(auto d=0; d<dispResolution; ++d){
//				double depth = dispToDepth(d);
//				Vector3d spt = cam.GetPosition() + ray * depth;
//				for(auto v=0; v<images.size(); ++v){
//					const theia::Camera cam2 = reconstruction.View(v + offset)->Camera();
//					Vector2d imgpt;
//					cam2.ProjectPoint(Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
//					imgpt = imgpt / downsample;
//					if(imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] >= width - 1 || imgpt[1] >= height - 1)
//						gv[d].push_back(0.0);
//					else {
//						VectorXd pix = interpolation_util::bilinear<uchar, 1>(grayImg[v].data, width, height, imgpt);
//						gv[d].push_back(pix[0]);
//					}
//
//					if(d == testDisp){
//						Mat outImg = grayImg[v].clone();
//						cvtColor(outImg, outImg, CV_GRAY2RGB);
//						circle(outImg, cv::Point(imgpt[0], imgpt[1]),1,cv::Scalar(0,0,255));
//						sprintf(buffer, "%s/temp/pattern_d%03d_v%03d.jpg", file_io.getDirectory().c_str(), d, v+offset);
//						imwrite(buffer, outImg);
//					}
//				}
//			}
//			sprintf(buffer, "%s/temp/pattern.txt", file_io.getDirectory().c_str());
//			ofstream fout(buffer);
//			CHECK(fout.is_open());
//			for(auto d=0; d<gv.size(); ++d){
//				for(auto v=0; v<gv[d].size(); ++v)
//					fout << gv[d][v] << ' ';
//				fout << endl;
//			}
		}

		//generate proposal
//		sprintf(buffer, "%s/temp/unarydisp_b%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		dispUnary.saveImage(string(buffer), 255.0 / (double)dispResolution);
//
//		Depth depthUnary;
//		depthUnary.initialize(width, height, -1);
//		for(auto i=0; i<width * height; ++i)
//			depthUnary[i] = dispToDepth(dispUnary[i]);
//		sprintf(buffer, "%s/temp/unaryDepth_b%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		depthUnary.saveImage(string(buffer), -1);


		//debug for SfM proposal
//		vector<Depth> SfMProposals;
//		ProposalSfM proposalSfM(file_io, images[anchor-offset], reconstruction, anchor, dispResolution, min_disp, max_disp, (double)downsample);
//		proposalSfM.genProposal(SfMProposals);


//		cout << "Solving with first order smoothness..." << endl;
//		FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), images[anchor-offset], MRF_data, (float)MRFRatio, dispResolution, (EnergyType)(MRFRatio * weight_smooth));
//		Depth result_firstOrder;
//		optimizer_firstorder.optimize(result_firstOrder, 10);
//		sprintf(buffer, "%s/temp/result%05d_firstorder_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
////		warpToAnchor(result_firstOrder, "firstorder");
//		result_firstOrder.saveImage(buffer, 255.0 / (double)dispResolution);
//
//		printf("Saving depth to point cloud...\n");
//		Depth depth_firstOrder;
//		disparityToDepth(result_firstOrder, depth_firstOrder);
//		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthToPly(string(buffer), depth_firstOrder, images[anchor-offset], reconstruction.View(anchor)->Camera(), downsample);

//		cout << "Solving with second order smoothness (trbp)..." << endl;
//		SecondOrderOptimizeTRBP optimizer_trbp(file_io, (int)images.size(), images[anchor-offset], MRF_data, (float)MRFRatio, dispResolution);
//		Depth result_trbp;
//		optimizer_trbp.optimize(result_trbp, 10);
//		sprintf(buffer, "%s/temp/result%05d_trbp_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_trbp.saveImage(buffer, 255.0 / (double)dispResolution);

		cout << "Solving with second order smoothness (fusion move)..." << endl;
		SecondOrderOptimizeFusionMove optimizer_fusion(file_io, (int)images.size(), images[anchor-offset], MRF_data, (float)MRFRatio, dispResolution, dispUnary, min_disp, max_disp);
		Depth result_fusion;
		optimizer_fusion.optimize(result_fusion, 1000);
		sprintf(buffer, "%s/temp/result%05d_fusionmove_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
		result_fusion.saveImage(buffer, 255.0 / (double)dispResolution);

		printf("Saving depth to point cloud...\n");
		Depth depth_fusion;
		disparityToDepth(result_fusion, depth_fusion);
		sprintf(buffer, "%s/temp/mesh_fusion_b%05d.ply", file_io.getDirectory().c_str(), anchor);
		utility::saveDepthToPly(string(buffer), depth_fusion, images[anchor-offset], reconstruction.View(anchor)->Camera(), downsample);

//		warpToAnchor(result_fusion, "fusion");
	}

	void DynamicStereo::warpToAnchor(const Depth& refDisp, const std::string& prefix) const{
		cout << "Warpping..." << endl;
		char buffer[1024] = {};

		vector<Mat> fullimages(images.size());
		for(auto i=0; i<fullimages.size(); ++i)
			fullimages[i] = imread(file_io.getImage(i+offset));
		vector<Mat> warpped(fullimages.size());

		const int w = fullimages[0].cols;
		const int h = fullimages[0].rows;

		const theia::Camera cam1 = reconstruction.View(anchor)->Camera();

		//in full resolution
		Mat warpMask;
		sprintf(buffer, "%s/mask%05d.jpg", file_io.getDirectory().c_str(), anchor);
		warpMask = imread(buffer);

		CHECK(warpMask.data) << "Empty mask";

		cvtColor(warpMask, warpMask, CV_RGB2GRAY);

		for(auto i=0; i<fullimages.size(); ++i){
			cout << i+offset << ' ' << flush;
			warpped[i] = fullimages[anchor-offset].clone();
			if(i == anchor-offset)
				continue;
			const theia::Camera cam2 = reconstruction.View(i+offset)->Camera();
			for(auto y=downsample; y<h-downsample; ++y){
				for(auto x=downsample; x<w-downsample; ++x){
					if(warpMask.at<uchar>(y,x) < 200)
						continue;
					Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x,y));
					ray.normalize();
					double disp = refDisp.getDepthAt(Vector2d(x/downsample, y/downsample));
					double depth = 1.0 / (disp / dispResolution * (max_disp-min_disp) + min_disp);
					Vector3d spt = cam1.GetPosition() + ray * depth;
					Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
					Vector2d imgpt;
					cam2.ProjectPoint(spt_homo, &imgpt);
					if(imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < w -1 && imgpt[1] < h - 1){
						Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(fullimages[i].data, w, h, imgpt);
						warpped[i].at<Vec3b>(y,x) = Vec3b(pix2[0], pix2[1], pix2[2]);
					}
				}
			}

		}

		cout << endl << "Saving..." << endl;

		for(auto i=0; i<warpped.size(); ++i){
			sprintf(buffer, "%s/temp/warpped_%s_b%05d_f%05d.jpg", file_io.getDirectory().c_str(),prefix.c_str(),  anchor, i+offset);
			imwrite(buffer, warpped[i]);
		}
	}

	void DynamicStereo::disparityToDepth(const Depth& disp, Depth& depth){
		depth.initialize(disp.getWidth(), disp.getHeight());
		for(auto i=0; i<disp.getWidth() * disp.getHeight(); ++i)
			depth[i] = dispToDepth(disp[i]);
	}
}
