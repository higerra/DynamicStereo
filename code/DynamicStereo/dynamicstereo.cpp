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

	namespace utility {
		void visualizeSegmentation(const std::vector<int> &labels, const int width, const int height, cv::Mat &output) {
			CHECK_EQ(width * height, labels.size());
			output = Mat(height, width, CV_8UC3);

			std::vector<cv::Vec3b> colorTable((size_t) width * height);
			std::default_random_engine generator;
			std::uniform_int_distribution<int> distribution(0, 255);

			for (int i = 0; i < width * height; i++) {
				for (int j = 0; j < 3; ++j)
					colorTable[i][j] = (uchar) distribution(generator);
			}

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++)
					output.at<cv::Vec3b>(y, x) = colorTable[labels[y * width + x]];
			}
		}


		void saveDepthAsPly(const string &path, const Depth &depth, const cv::Mat &image, const theia::Camera &cam,
							const int downsample) {
			CHECK_EQ(depth.getWidth(), image.cols);
			CHECK_EQ(depth.getHeight(), image.rows);
			TriMesh mesh;
			mesh.request_vertex_colors();
			const int &w = depth.getWidth();
			const int &h = depth.getHeight();

			vector<TriMesh::VertexHandle> vhandle((size_t) (w * h));
			Vector3d camcenter = cam.GetPosition();
			TriMesh::VertexHandle ch = mesh.add_vertex(TriMesh::Point(camcenter[0], camcenter[1], camcenter[2]));
			mesh.set_color(ch, TriMesh::Color(255, 0, 0));

			int vid = 0;
			for (auto y = 0; y < h; ++y) {
				for (auto x = 0; x < w; ++x, ++vid) {
					if(depth(x,y) <= 0)
						continue;
					cv::Vec3b pix = image.at<Vec3b>(y, x);
					Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(x * downsample, y * downsample));
					//ray.normalize();
					Vector3d spt = cam.GetPosition() + ray * depth(x, y);
					vhandle[vid] = mesh.add_vertex(TriMesh::Point(spt[0], spt[1], spt[2]));
					mesh.set_color(vhandle[vid], TriMesh::Color(pix[2], pix[1], pix[0]));
				}
			}

			OpenMesh::IO::Options wopt;
			wopt += OpenMesh::IO::Options::VertexColor;
			CHECK(OpenMesh::IO::write_mesh(mesh, path, wopt)) << "Can not write ply file " << path;
		}

		void stereoSGBM(const cv::Mat &img1, const cv::Mat &img2) {
		}

	}

	DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
	                             const int tWindow_, const int tWindowStereo_, const int downsample_, const double weight_smooth_, const int dispResolution_,
	                             const double min_disp_, const double max_disp_):
			file_io(file_io_), anchor(anchor_), tWindow(tWindow_), tWindowStereo(tWindowStereo_), downsample(downsample_), dispResolution(dispResolution_), pR(3),
			min_disp(min_disp_), max_disp(max_disp_){
		CHECK_LE(tWindowStereo, tWindow);
		cout << "Reading..." << endl;
		offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
		CHECK_GE(file_io.getTotalNum(), offset + tWindow);
		cout << "Reading reconstruction" << endl;
		CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Can not open reconstruction file";
		CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());

		const vector<theia::ViewId>& vids = reconstruction.ViewIds();
		orderedId.resize(vids.size());
		for(auto i=0; i<vids.size(); ++i) {
			const theia::View* v = reconstruction.View(vids[i]);
			std::string nstr = v->Name().substr(5,5);
			int idx = atoi(nstr.c_str());
			orderedId[i] = IdPair(idx, vids[i]);
		}
		std::sort(orderedId.begin(), orderedId.end(),
				  [](const std::pair<int, theia::ViewId>& v1, const std::pair<int, theia::ViewId>& v2){return v1.first < v2.first;});

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

			const theia::Camera cam = reconstruction.View(orderedId[i+offset].second)->Camera();
			cout << "Projection matrix for view " << orderedId[i+offset].first<< endl;
			theia::Matrix3x4d pm;
			cam.GetProjectionMatrix(&pm);
			cout << pm << endl;
			double dis1 = cam.RadialDistortion1();
			double dis2 = cam.RadialDistortion2();
			printf("Radio distortion:(%.5ff,%.5f)\n", dis1, dis2);
		}
		CHECK_GT(images.size(), 2) << "Too few images";
		width = images.front().cols;
		height = images.front().rows;
		dispUnary.initialize(width, height, 0.0);

		model = shared_ptr<StereoModel<EnergyType> >(new StereoModel<EnergyType>(images[anchor-offset], (double)downsample, dispResolution, 1000, weight_smooth_));
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

		theia::Camera cam1 = reconstruction.View(orderedId[id1].second)->Camera();
		theia::Camera cam2 = reconstruction.View(orderedId[id2].second)->Camera();

		Vector3d ray1 = cam1.PixelToUnitDepthRay(pt);
		//ray1.normalize();
		imgL = imread(file_io.getImage(id1));
		imgR = imread(file_io.getImage(id2));
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
			imgpt = imgpt;
			//printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
			cv::Point cvpt(((int)imgpt[0]), ((int)imgpt[1]));
			cv::circle(imgR, cvpt, 1, cv::Scalar(255 - cindex * 255.0, 0 ,cindex * 255.0));
			cindex += 1.0 / steps;
		}
	}



	void DynamicStereo::runStereo() {
		char buffer[1024] = {};

		initMRF();

		{
			//debug: inspect unary term
			const int tx = 1078 / downsample;
			const int ty = 258 / downsample;
			printf("Unary term for (%d,%d)\n", tx, ty);
			for (auto d = 0; d < dispResolution; ++d) {
				cout << model->operator()(ty * width + tx, d) << ' ';
			}
			cout << endl;
			printf("noisyDisp(%d,%d): %.2f\n", tx, ty, dispUnary.getDepthAtInt(tx, ty));

			const theia::Camera &cam = reconstruction.View(orderedId[anchor].second)->Camera();
			Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(tx * downsample, ty * downsample));
			//ray.normalize();

			int tdisp = (int) dispUnary(tx, ty);
			//int tdisp = 30;
			double td = model->dispToDepth(tdisp);
			printf("Cost at d=%d: %d\n", tdisp, model->operator()(ty * width + tx, tdisp));

			Vector3d spt = cam.GetPosition() + ray * td;
			for (auto v = 0; v < images.size(); ++v) {
				Mat curimg = imread(file_io.getImage(v + offset));
				Vector2d imgpt;
				reconstruction.View(orderedId[v + offset].second)->Camera().ProjectPoint(
						Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
				if (imgpt[0] >= 0 || imgpt[1] >= 0 || imgpt[0] < width || imgpt[1] < height)
					cv::circle(curimg, cv::Point(imgpt[0], imgpt[1]), 2, cv::Scalar(0, 0, 255), 2);
				sprintf(buffer, "%s/temp/project_b%05d_v%05d.jpg", file_io.getDirectory().c_str(), anchor,
						v + offset);
				imwrite(buffer, curimg);
			}
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
//		dispUnary.saveImage(string(buffer), 255.0 / (double) dispResolution);
//
//
//
////
//		Depth depthUnary;
//		depthUnary.initialize(width, height, -1);
//		for(auto i=0; i<width * height; ++i)
//			depthUnary[i] = model->dispToDepth(dispUnary[i]);
//		sprintf(buffer, "%s/temp/unaryDepth_b%05d.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depthUnary, images[anchor-offset], reconstruction.View(orderedId[anchor].second)->Camera(), downsample);


		//debug for SfM proposal
//		vector<Depth> SfMProposals;
//		ProposalSfM proposalSfM(file_io, images[anchor-offset], reconstruction, anchor, dispResolution, min_disp, max_disp, (double)downsample);
//		proposalSfM.genProposal(SfMProposals);


		cout << "Solving with first order smoothness..." << endl;
		FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), model);
		Depth result_firstOrder;
		optimizer_firstorder.optimize(result_firstOrder, 10);
		//sprintf(buffer, "%s/temp/result%05d_firstorder_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
		warpToAnchor(result_firstOrder, "firstorder");
		//result_firstOrder.saveImage(buffer, 255.0 / (double)dispResolution);

		printf("Saving depth to point cloud...\n");
		Depth depth_firstOrder;
		disparityToDepth(result_firstOrder, depth_firstOrder);
		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d.ply", file_io.getDirectory().c_str(), anchor);
		utility::saveDepthAsPly(string(buffer), depth_firstOrder, images[anchor-offset], reconstruction.View(orderedId[anchor].second)->Camera(), downsample);

		Depth disp_firstOrder_filtered, depth_firstOrder_filtered;
		printf("Applying bilateral filter to depth:\n");
		bilateralFilter(result_firstOrder, images[anchor-offset], disp_firstOrder_filtered, 21, 5, 30, 3);
		disparityToDepth(disp_firstOrder_filtered, depth_firstOrder_filtered);
		depth_firstOrder_filtered.updateStatics();
		sprintf(buffer, "%s/temp/mesh_firstorder_b%05d_filtered.ply", file_io.getDirectory().c_str(), anchor);
		utility::saveDepthAsPly(string(buffer), depth_firstOrder_filtered, images[anchor-offset], reconstruction.View(orderedId[anchor].second)->Camera(), downsample);
		warpToAnchor(disp_firstOrder_filtered, "firstorder_bifiltered");
//
//		{
//			//test for GridWarpping
//			vector<Mat> fullImg(images.size());
//			for (auto i = 0; i < fullImg.size(); ++i)
//				fullImg[i] = imread(file_io.getImage(i + offset));
//			GridWarpping gridWarpping(file_io, anchor, fullImg, *model, reconstruction, orderedId,
//			                          depth_firstOrder_filtered, downsample, offset);
//
//			printf("================\nTesting for bilinear coefficience\n");
//			Vector2d testP(500, 500);
//			Vector4i testInd;
//			Vector4d testW;
//			gridWarpping.getGridIndAndWeight(testP, testInd, testW);
//			printf("(%d,%d,%d,%d), (%.2f,%.2f,%.2f,%.2f)\n", testInd[0], testInd[1], testInd[2], testInd[3],
//			       testW[0], testW[1], testW[2], testW[3]);
//
//			Mat mask;
//			sprintf(buffer, "%s/mask%05d.jpg", file_io.getDirectory().c_str(), anchor);
//			mask = imread(buffer);
//			CHECK(mask.data);
//			CHECK_EQ(mask.cols, fullImg[0].cols);
//			CHECK_EQ(mask.rows, fullImg[0].rows);
//			cvtColor(mask, mask, CV_RGB2GRAY);
//
//			for (auto i = 0; i < fullImg.size(); ++i) {
//				printf("=================\nWarpping frame %d\n", i);
//				vector<Vector2d> refPt, srcPt;
//				const int testF = i;
//				printf("Computing point correspondence...\n");
//				//gridWarpping.computePointCorrespondence(testF, refPt, srcPt);
//				gridWarpping.computePointCorrespondenceNoWarp(testF, refPt, srcPt);
//				CHECK_EQ(refPt.size(), srcPt.size());
//
//				printf("Done, correspondence: %d\n", (int) refPt.size());
//
//
//
//				Mat warpped = fullImg[anchor-offset].clone();
//				const theia::Camera &refCam = reconstruction.View(orderedId[anchor].second)->Camera();
//				const theia::Camera &srcCam = reconstruction.View(orderedId[testF + offset].second)->Camera();
//				for (auto y = downsample; y < warpped.rows - downsample; ++y) {
//					for (auto x = downsample; x < warpped.cols - downsample; ++x) {
//						if(mask.at<uchar>(y,x) < 200)
//							continue;
//						double d = depth_firstOrder_filtered.getDepthAt(
//								Vector2d((double) x / (double) downsample, (double) y / (double) downsample));
//						Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x, y));
//						Vector3d spt = refCam.GetPosition() + d * ray;
//						Vector2d imgpt;
//						srcCam.ProjectPoint(spt.homogeneous(), &imgpt);
//						if (imgpt[0] < 0 || imgpt[0] > warpped.cols - 1 || imgpt[1] < 0 || imgpt[1] > warpped.rows - 1)
//							continue;
//						Vector3d pix = interpolation_util::bilinear<uchar, 3>(fullImg[testF].data, warpped.cols,
//						                                                      warpped.rows, imgpt);
//						warpped.at<Vec3b>(y, x) = Vec3b((uchar) pix[0], (uchar) pix[1], (uchar) pix[2]);
//					}
//				}
//				sprintf(buffer, "%s/temp/stereo%05d.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, warpped);
//				Mat grayRef, graySrc, colorRef, colorSrc;
//				colorSrc = fullImg[testF].clone();
//				colorRef = fullImg[anchor-offset].clone();
////				cvtColor(fullImg[anchor - offset], grayRef, CV_RGB2GRAY);
////				cvtColor(warpped, graySrc, CV_RGB2GRAY);
////
////				cvtColor(grayRef, colorRef, CV_GRAY2RGB);
////				cvtColor(graySrc, colorSrc, CV_GRAY2RGB);
//
////				for (auto x = 0; x < fullImg[0].cols; x += gridWarpping.getBlockW()) {
////					cv::line(colorRef, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////					cv::line(colorSrc, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////				}
////				for (auto y = 0; y < fullImg[0].rows; y += gridWarpping.getBlockH()) {
////					cv::line(colorRef, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
////					cv::line(colorSrc, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
////				}
//				Mat tgtImg = fullImg[testF].clone();
//				drawKeyPoints(tgtImg, srcPt);
//				sprintf(buffer, "%s/temp/trackOnTgt%05d.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, tgtImg);
//
//				Mat refImg = fullImg[anchor-offset].clone();
//				drawKeyPoints(refImg, refPt);
//				sprintf(buffer, "%s/temp/trackOnRef%05d.jpg", file_io.getDirectory().c_str(), anchor-offset);
//				imwrite(buffer, refImg);
//
//				drawKeyPoints(colorRef, refPt);
//
//				Mat stabled, vis;
//				Mat comb;
//				gridWarpping.computeWarppingField(testF, refPt, srcPt, fullImg[testF], stabled, vis, true);
//
////				hconcat(stabled, vis, comb);
////				sprintf(buffer, "%s/temp/sta_%05dimg1.jpg", file_io.getDirectory().c_str(), testF);
////				imwrite(buffer, colorRef);
////				sprintf(buffer, "%s/temp/sta_%05dimg2.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, colorSrc);
////				sprintf(buffer, "%s/temp/sta_%05dimg3.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, colorRef);
////				sprintf(buffer, "%s/temp/sta_%05dimg3.jpg", file_io.getDirectory().c_str(), testF);
////				imwrite(buffer, colorRef);
//
//
//				for(auto y=0; y<stabled.rows; ++y){
//					for(auto x=0; x<stabled.cols; ++x){
//						if(mask.at<uchar>(y,x) < 200)
//							stabled.at<Vec3b>(y,x) = fullImg[anchor-offset].at<Vec3b>(y,x);
//					}
//				}
//
//				sprintf(buffer, "%s/temp/sta_%05dimg4.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, stabled);
//
////				sprintf(buffer, "%s/temp/unstabled%05d.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, warpped);
////
////				sprintf(buffer, "%s/temp/stabbled%05d.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, stabled);
////			sprintf(buffer, "%s/temp/sta_gri%05d.jpg", file_io.getDirectory().c_str(), testF+offset);
////			imwrite(buffer, vis);
////			sprintf(buffer, "%s/temp/sta_com%05d.jpg", file_io.getDirectory().c_str(), testF+offset);
////			imwrite(buffer, comb);
//			}
//		}

//		cout << "Solving with second order smoothness (trbp)..." << endl;
//		SecondOrderOptimizeTRBP optimizer_trbp(file_io, (int)images.size(), model);
//		Depth result_trbp, depth_trbp;
//		optimizer_trbp.optimize(result_trbp, 10);
//		disparityToDepth(result_trbp, depth_trbp);
//
//		sprintf(buffer, "%s/temp/result%05d_trbp_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_trbp.saveImage(buffer, 255.0 / (double)dispResolution);
//		sprintf(buffer, "%s/temp/mesh%05d_trbp.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_trbp, images[anchor-offset], reconstruction.View(orderedId[anchor].second)->Camera(), downsample);


//		cout << "Solving with second order smoothness (fusion move)..." << endl;
//		SecondOrderOptimizeFusionMove optimizer_fusion(file_io, images.size(), model, dispUnary);
//		const vector<int>& refSeg = optimizer_fusion.getRefSeg();
//		Mat segImg;
//		utility::visualizeSegmentation(refSeg, width, height, segImg);
//		sprintf(buffer, "%s/temp/refSeg%.5d.jpg", file_io.getDirectory().c_str(), anchor);
//		imwrite(buffer, segImg);
//		Depth result_fusion;
//		optimizer_fusion.optimize(result_fusion, 300);
//
//		sprintf(buffer, "%s/temp/result%05d_fusionmove_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_fusion.saveImage(buffer, 255.0 / (double)dispResolution);
//		printf("Saving depth to point cloud...\n");
//		Depth depth_fusion;
//		disparityToDepth(result_fusion, depth_fusion);
//		sprintf(buffer, "%s/temp/mesh_fusion_b%05d.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_fusion, images[anchor-offset], reconstruction.View(anchor)->Camera(), downsample);
//		warpToAnchor(result_fusion, "fusion");

//		cout << "Solving with second order smoothness (TRWS)..." << endl;
//		SecondOrderOptimizeTRWS optimizer_TRWS(file_io, (int)images.size(), model);
//		Depth result_TRWS;
//		optimizer_TRWS.optimize(result_TRWS, 1);
//
//		sprintf(buffer, "%s/temp/result%05d_TRWS_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_TRWS.saveImage(buffer, 255.0 / (double)dispResolution);
//		warpToAnchor(result_TRWS, "TRWS");
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

		const theia::Camera cam1 = reconstruction.View(orderedId[anchor].second)->Camera();

		//in full resolution
		Mat warpMask;
		sprintf(buffer, "%s/mask%05d.jpg", file_io.getDirectory().c_str(), anchor);
		warpMask = imread(buffer);
//		Mat warpMask(h,w,CV_8UC3, Scalar(255,255,255));

		CHECK(warpMask.data) << "Empty mask";

		cvtColor(warpMask, warpMask, CV_RGB2GRAY);

		int startid = anchor - offset;
		int endid = anchor - offset;

		for(auto i=0; i<fullimages.size(); ++i){
			cout << i+offset << ' ' << flush;
			warpped[i] = fullimages[anchor-offset].clone();
			if(i == anchor-offset)
				continue;
			bool invalid = false;
			const theia::Camera cam2 = reconstruction.View(orderedId[i+offset].second)->Camera();
			for(auto y=downsample; y<h-downsample; ++y){
				for(auto x=downsample; x<w-downsample; ++x){
					if(warpMask.at<uchar>(y,x) < 200)
						continue;
					Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x,y));
					//ray.normalize();
					double disp = refDisp.getDepthAt(Vector2d(x/downsample, y/downsample));
					Vector3d spt = cam1.GetPosition() + ray * model->dispToDepth(disp);
					Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
					Vector2d imgpt;
					cam2.ProjectPoint(spt_homo, &imgpt);
					if(imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < w -1 && imgpt[1] < h - 1){
						Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(fullimages[i].data, w, h, imgpt);
						warpped[i].at<Vec3b>(y,x) = Vec3b(pix2[0], pix2[1], pix2[2]);
					}else{
						warpped[i].at<Vec3b>(y,x) = Vec3b(0,0,0);
						invalid = true;
						break;
					}
				}
				if(invalid) {
					break;
				}
			}
			if(!invalid){
				endid = std::max(endid, i);
				startid = std::min(startid, i);
			}
		}
		cout << endl;

		//applying a median filter
		vector<Mat> oriWarpped(warpped.size());
		for(auto i=startid; i<=endid; ++i)
			oriWarpped[i] = warpped[i].clone();
//		const int r = 5;
//		printf("Applying median filter, r = %d\n", r);
//		for(auto i=startid; i<=endid; ++i) {
//			int s, e;
//			if (i - r < startid) {
//				s = startid;
//				e = startid + 2 * r + 1;
//			}
//			else if (i + r > endid) {
//				s = endid - 2 * r - 1;
//				e = endid;
//			} else {
//				s = i - r;
//				e = i + r;
//			}
//			for(auto y=0; y<h; ++y){
//				for(auto x=0; x<w; ++x){
//					if(warpMask.at<uchar>(y,x) < 200)
//						continue;
//					vector<int> rc,gc,bc;
//					for(auto t = s; t <= e; ++t){
//						Vec3b pix = oriWarpped[t].at<Vec3b>(y,x);
//						rc.push_back(pix[0]);
//						gc.push_back(pix[1]);
//						bc.push_back(pix[2]);
//					}
//					nth_element(rc.begin(), rc.begin()+r, rc.end());
//					nth_element(gc.begin(), gc.begin()+r, gc.end());
//					nth_element(bc.begin(), bc.begin()+r, bc.end());
//					warpped[i].at<Vec3b>(y,x) = Vec3b((uchar)rc[r], (uchar)gc[r], (uchar)bc[r]);
//				}
//			}
//		}

		printf("Saving: start: %d, end:%d\n", startid, endid);
		for(auto i=startid; i<=endid; ++i){
			sprintf(buffer, "%s/temp/warpped_%s_b%05d_f%05d.jpg", file_io.getDirectory().c_str(),prefix.c_str(),  anchor, i+offset);
			imwrite(buffer, warpped[i]);
		}
	}

	void DynamicStereo::disparityToDepth(const Depth& disp, Depth& depth){
		depth.initialize(disp.getWidth(), disp.getHeight(), -1);
		for(auto i=0; i<disp.getWidth() * disp.getHeight(); ++i) {
			if(disp[i] <= 0) {
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
