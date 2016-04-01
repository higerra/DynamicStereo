//
// Created by yanhang on 3/28/16.
//

#include "gridwarpping.h"
#include "gridenergy.h"
#include <random>
using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
	GridWarpping::GridWarpping(const FileIO &file_io_, const int anchor_, const std::vector<cv::Mat> &images_,
	                           const StereoModel<EnergyType> &model_,
	                           const theia::Reconstruction &reconstruction_, const OrderedIDSet& orderedSet_, Depth &refDepth_,
	                           const int downsample_, const int offset_,
	                           const int gw, const int gh) :
			file_io(file_io_), anchor(anchor_), images(images_), model(model_), reconstruction(reconstruction_), orderedId(orderedSet_),refDepth(refDepth_),
			offset(offset_), downsample(downsample_), gridW(gw), gridH(gh) {
		CHECK(!images.empty());
		width = images[0].cols;
		height = images[0].rows;
		CHECK_EQ(width, model.width * downsample);
		CHECK_EQ(height, model.height * downsample);
		blockW = (double) width / gridW;
		blockH = (double) height / gridH;
		gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
		for (auto x = 0; x <= gridW; ++x) {
			for (auto y = 0; y <= gridH; ++y) {
				gridLoc[y * (gridW + 1) + x] = Eigen::Vector2d(blockW * x, blockH * y);
				if (x == gridW)
					gridLoc[y * (gridW + 1) + x][0] -= 1.1;
				if (y == gridH)
					gridLoc[y * (gridW + 1) + x][1] -= 1.1;
			}
		}
	}

	void GridWarpping::visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat &img) const {
		CHECK_EQ(grid.size(), gridLoc.size());
//		Mat oriGrid = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		img = Mat(height, width, CV_8UC3, Scalar(0,0,0));
//		for (auto x = 0; x < width; x += getBlockW()) {
//			cv::line(oriGrid, cv::Point(x, 0), cv::Point(x, height - 1), Scalar(255, 255, 255), 1);
//		}
//		for (auto y = 0; y < height; y += getBlockH()) {
//			cv::line(oriGrid, cv::Point(0, y), cv::Point(width - 1, y), Scalar(255, 255, 255), 1);
//		}
//		for (auto y = 0; y < height; ++y) {
//			for (auto x = 0; x < width; ++x) {
//				Vector4i ind;
//				Vector4d w;
//				getGridIndAndWeight(Vector2d(x, y), ind, w);
//				Vector2d pt(0, 0);
//				for (auto i = 0; i < 4; ++i) {
//					pt += grid[ind[i]] * w[i];
//				}
//				if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
//					continue;
//				Vector3d pixG = interpolation_util::bilinear<uchar, 3>(oriGrid.data, oriGrid.cols, oriGrid.rows, pt);
//				img.at<Vec3b>(y, x) = Vec3b((uchar) pixG[0], (uchar) pixG[1], (uchar) pixG[2]);
//			}
//		}
		for(auto gy=0; gy<gridH; ++gy) {
			for (auto gx = 0; gx < gridW; ++gx){
				const int gid1 = gy * (gridW+1) + gx;
				const int gid2 = (gy+1) * (gridW+1) + gx;
				const int gid3 = (gy+1)*(gridW+1)+gx+1;
				const int gid4= gy * (gridW+1) + gx+1;
				cv::line(img, cv::Point(grid[gid1][0], grid[gid1][1]), cv::Point(grid[gid2][0], grid[gid2][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid2][0], grid[gid2][1]), cv::Point(grid[gid3][0], grid[gid3][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid3][0], grid[gid3][1]), cv::Point(grid[gid4][0], grid[gid4][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid4][0], grid[gid4][1]), cv::Point(grid[gid1][0], grid[gid1][1]), Scalar(255,255,255));
			}
		}
	}

	void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
	                                       Eigen::Vector4d &w) const {
		CHECK_LE(pt[0], width - 1);
		CHECK_LE(pt[1], height - 1);
		int x = (int) floor(pt[0] / blockW);
		int y = (int) floor(pt[1] / blockH);

		//////////////
		// 1--2
		// |  |
		// 4--3
		/////////////
		ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
					   (y + 1) * (gridW + 1) + x);

		const double &xd = pt[0];
		const double &yd = pt[1];
		const double xl = gridLoc[ind[0]][0];
		const double xh = gridLoc[ind[2]][0];
		const double yl = gridLoc[ind[0]][1];
		const double yh = gridLoc[ind[2]][1];

		w[0] = (xh - xd) * (yh - yd);
		w[1] = (xd - xl) * (yh - yd);
		w[2] = (xd - xl) * (yd - yl);
		w[3] = (xh - xd) * (yd - yl);

		double s = w[0] + w[1] + w[2] + w[3];
		CHECK_GT(s, 0) << pt[0] << ' '<< pt[1];
		w = w / s;

		Vector2d pt2 =
				gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
		double error = (pt2 - pt).norm();
		CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
	}

	void GridWarpping::computePointCorrespondence(const int id, std::vector<Eigen::Vector2d> &refPt,
	                                              std::vector<Eigen::Vector2d> &srcPt) const {
		const vector<theia::TrackId> trackIds = reconstruction.TrackIds();
		//tracks that are visible on both frames
		printf("Collecting visible points\n");
		vector<theia::TrackId> visiblePts;
		for (auto tid: trackIds) {
			const theia::Track *t = reconstruction.Track(tid);
			const std::unordered_set<theia::ViewId> &viewids = t->ViewIds();
			if ((viewids.find(orderedId[anchor].second) != viewids.end()) &&
			    (viewids.find(orderedId[id + offset].second) != viewids.end()))
				visiblePts.push_back(tid);
		}

		//transfer depth from reference view to source view
		printf("Transfering depth from %d to %d\n", anchor, id+offset);
		const theia::Camera& refCam = reconstruction.View(orderedId[anchor].second)->Camera();
		const theia::Camera& srcCam = reconstruction.View(orderedId[id + offset].second)->Camera();

		Depth zBuffer(width / 4, height / 4, numeric_limits<double>::max());
		vector<vector<list < Vector3d> > > hTable(width);
		for (auto &ht: hTable)
			ht.resize(height);
		double zMargin = 0.1 * refDepth.getMedianDepth();
		for (auto y = 0; y < height-downsample; ++y) {
			for (auto x = 0; x < width-downsample; ++x) {
				Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x, y));
				Matrix<double,1,1> dRef = interpolation_util::bilinear<double, 1>(refDepth.getRawData().data(), refDepth.getWidth(),
				                                                                  refDepth.getHeight(), Vector2d((double)x/downsample, (double)y/downsample));
				Vector3d spt = refCam.GetPosition() + dRef(0,0) * ray;
				Vector2d imgpt;
				double d = srcCam.ProjectPoint(spt.homogeneous(), &imgpt);
				if (imgpt[0] >= 0 && imgpt[0] < width - 1 && imgpt[1] >= 0 && imgpt[1] < height - 1) {
					const int flx = (int) floor(imgpt[0]);
					const int fly = (int) floor(imgpt[1]);
					if (d < zBuffer(flx / 4, fly / 4) + zMargin) {
						hTable[flx][fly].push_back(Vector3d(imgpt[0], imgpt[1], d));
						zBuffer(flx/4, fly/4) = std::min(d, zBuffer(flx/4, fly/4));
					}
				}
			}
		}

		printf("Re-warping points\n");
		printf("Number of visible structure points: %d\n", (int)visiblePts.size());
		const int nR = 2;
		const double sigma = (double) nR * 0.5;
		for (auto tid: visiblePts) {
			Vector2d ptRef, ptSrc;
			const Vector4d &spt = reconstruction.Track(tid)->Point();
			double dRef = refCam.ProjectPoint(spt, &ptRef);
			double dSrc = srcCam.ProjectPoint(spt, &ptSrc);
			if (ptRef[0] < 0 || ptRef[1] < 0 || ptRef[0] > width - 1 || ptRef[1] > height - 1)
				continue;
			if (ptSrc[0] < 0 || ptSrc[1] < 0 || ptSrc[0] > width - 1 || ptSrc[1] > height - 1)
				continue;

			const int flxsrc = (int) floor(ptSrc[0]);
			const int flysrc = (int) floor(ptSrc[1]);
			//interpolate depth by near by depth samples.
			//Note, interpolation should be applied to inverse depth
			double dsrc2 = 0.0;
			double w = 0.0;
			for (auto dx = -1 * nR; dx <= nR; ++dx) {
				for (auto dy = -1 * nR; dy <= nR; ++dy) {
					int curx = flxsrc + dx;
					int cury = flysrc + dy;
					if (curx < 0 || curx > width - 1 || cury < 0 || cury > height - 1)
						continue;
					for (auto sample: hTable[curx][cury]) {
						CHECK_GT(sample[2], 0.0);
						Vector2d off = ptSrc - Vector2d(sample[0], sample[1]);
						double curw = math_util::gaussian(0, sigma, off.norm());
						dsrc2 += (1.0 / sample[2]) * curw;
						w += curw;
					}
				}
			}
			if(w == 0) {
				continue;
			}
			dsrc2 /= w;
			dsrc2 = 1.0 / dsrc2;
			Vector3d sptSrc = dsrc2 * srcCam.PixelToUnitDepthRay(ptSrc) + srcCam.GetPosition();

			Vector2d ptRef2;
			refCam.ProjectPoint(sptSrc.homogeneous(), &ptRef2);
			if (ptRef2[0] < 0 || ptRef2[0] > width - 1 || ptRef2[1] < 0 || ptRef2[1] > height - 1)
				continue;
			refPt.push_back(ptRef);
			srcPt.push_back(ptRef2);
		}
	}

	void GridWarpping::computePointCorrespondenceNoWarp(const int id, std::vector<Eigen::Vector2d> &refPt,
	                                      std::vector<Eigen::Vector2d> &srcPt) const{
		const vector<theia::TrackId> trackIds = reconstruction.TrackIds();
		//tracks that are visible on both frames
		const theia::Camera& refCam = reconstruction.View(orderedId[anchor].second)->Camera();
		const theia::Camera& srcCam = reconstruction.View(orderedId[id + offset].second)->Camera();

		for (auto tid: trackIds) {
			const theia::Track *t = reconstruction.Track(tid);
			const std::unordered_set<theia::ViewId> &viewids = t->ViewIds();
			if ((viewids.find(orderedId[anchor].second) == viewids.end()) ||
			    (viewids.find(orderedId[id + offset].second) == viewids.end()))
				continue;
			Vector2d imgptRef;
			Vector2d imgptSrc;
			refCam.ProjectPoint(t->Point(), &imgptRef);
			srcCam.ProjectPoint(t->Point(), &imgptSrc);
			refPt.push_back(imgptRef);
			srcPt.push_back(imgptSrc);
		}
	}




	void GridWarpping::computeWarppingField(const int id, const std::vector<Eigen::Vector2d> &refPt,
	                                        const std::vector<Eigen::Vector2d> &srcPt,
											const cv::Mat& inputImg, cv::Mat& outputImg, cv::Mat &vis,
											const bool initByStereo) const{
		CHECK_EQ(refPt.size(), srcPt.size());
		CHECK_EQ(inputImg.cols, width);
		CHECK_EQ(inputImg.rows, height);
		char buffer[1024] = {};
		vector<vector<double> > vars(gridLoc.size());
		for (auto &v: vars)
			v.resize(2);

		const theia::Camera& refCam = reconstruction.View(orderedId[anchor].second)->Camera();
		const theia::Camera& srcCam = reconstruction.View(orderedId[id + offset].second)->Camera();

		vector<Vector2d> grid2(gridLoc.size());
		for (auto i = 0; i < gridLoc.size(); ++i) {
			if(initByStereo){
				Vector2d imgpt = gridLoc[i] / downsample;
				if(imgpt[0] >= model.width - 1)
					imgpt[0] = model.width - 1.1;
				if(imgpt[1] >= model.height - 1)
					imgpt[1] = model.height - 1.1;
				double d = refDepth.getDepthAt(imgpt);
				Vector3d spt = refCam.GetPosition() + d * refCam.PixelToUnitDepthRay(gridLoc[i]);
				Vector2d imgptSrc;
				srcCam.ProjectPoint(spt.homogeneous(), &imgptSrc);
				vars[i][0] = imgptSrc[0];
				vars[i][1] = imgptSrc[1];
				grid2[i] = imgptSrc;
			}else{
				vars[i][0] = gridLoc[i][0];
				vars[i][1] = gridLoc[i][1];
				grid2[i] = gridLoc[i];
			}
		}

		vector<Vector2d> refPt2(refPt.size(), Vector2d(0,0));
		for(auto i=0; i<refPt.size(); ++i){
			Vector4i ind;
			Vector4d w;
			getGridIndAndWeight(refPt[i], ind, w);
			for(auto j=0; j<4; ++j)
				refPt2[i] += grid2[ind[j]] * w[j];
		}

		Mat initGrid, initWarp;
		visualizeGrid(grid2, initGrid);
		initWarp = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		for (auto y = 0; y < height; ++y) {
			for (auto x = 0; x < width; ++x) {
				Vector4i ind;
				Vector4d w;
				getGridIndAndWeight(Vector2d(x, y), ind, w);
				Vector2d pt(0, 0);
				for (auto i = 0; i < 4; ++i) {
					pt += grid2[ind[i]] * w[i];
				}
				if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
					continue;
				Vector3d pixW = interpolation_util::bilinear<uchar, 3>(inputImg.data, inputImg.cols, inputImg.rows, pt);
				initWarp.at<Vec3b>(y, x) = Vec3b((uchar) pixW[0], (uchar) pixW[1], (uchar) pixW[2]);
			}
		}
		drawKeyPoints(initGrid, refPt2);
		Mat inputImg3 = inputImg.clone();
		drawKeyPoints(inputImg3, srcPt);

		sprintf(buffer, "%s/temp/refPt%05d_1.jpg", file_io.getDirectory().c_str(), id);
		imwrite(buffer, initGrid);

		Mat srcPtGT;
		visualizeGrid(gridLoc, srcPtGT);
		drawKeyPoints(srcPtGT, srcPt);
		sprintf(buffer, "%s/temp/refPt%05d_3.jpg", file_io.getDirectory().c_str(), id);
		imwrite(buffer, srcPtGT);
//
//		sprintf(buffer, "%s/temp/sta_%05dimg2.jpg", file_io.getDirectory().c_str(), id);
//		imwrite(buffer, initWarp);

		ceres::Problem problem;
		printf("Creating problem...\n");
		const double wdata = 1.0;
		const double wregular = 0.01;

		const double truncDis = 20;
		for (auto i = 0; i < refPt.size(); ++i) {
			double dis = (refPt2[i] - srcPt[i]).norm();
			if(dis > truncDis)
				continue;
			Vector4i indRef;
			Vector4d bwRef;
			getGridIndAndWeight(refPt[i], indRef, bwRef);
			problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<WarpFunctorData, 1, 2, 2, 2, 2>(new WarpFunctorData(srcPt[i], bwRef, wdata)),
					new ceres::HuberLoss(5),
					vars[indRef[0]].data(), vars[indRef[1]].data(), vars[indRef[2]].data(), vars[indRef[3]].data());
		}


//		for (auto i = 0; i < gridLoc.size(); ++i)
//			problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WarpFunctorRegularization, 1, 2>(
//					new WarpFunctorRegularization(grid2[i], wregular)), NULL, vars[i].data());

		double wsimilarity = 0.0001;
		//similarity term
		for(auto y=1; y<=gridH; ++y) {
			for (auto x = 0; x < gridW; ++x) {
				int gid1, gid2, gid3;
				gid1 = y * (gridW + 1) + x;
				gid2 = (y - 1) * (gridW + 1) + x;
				gid3 = y * (gridW + 1) + x + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
						new WarpFunctorSimilarity(gridLoc[gid1], gridLoc[gid2], gridLoc[gid3], wsimilarity)), new ceres::HuberLoss(5),
										 vars[gid1].data(), vars[gid2].data(), vars[gid3].data());
				gid2 = (y - 1) * (gridW + 1) + x+1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
						new WarpFunctorSimilarity(gridLoc[gid1], gridLoc[gid2], gridLoc[gid3], wsimilarity)), new ceres::HuberLoss(5),
										 vars[gid1].data(), vars[gid2].data(), vars[gid3].data());
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		//options.minimizer_progress_to_stdout = true;

		ceres::Solver::Summary summary;
		printf("Solving...\n");
		ceres::Solve(options, &problem, &summary);


		cout << summary.BriefReport() << endl;

		outputImg = Mat(height, width, CV_8UC3, Scalar(0, 0, 0));
		vis = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		Mat oriGrid = inputImg.clone();

		printf("Warpping...\n");
		for (auto y = 0; y < height; ++y) {
			for (auto x = 0; x < width; ++x) {
				Vector4i ind;
				Vector4d w;
				getGridIndAndWeight(Vector2d(x, y), ind, w);
				Vector2d pt(0, 0);
				for (auto i = 0; i < 4; ++i) {
					pt[0] += vars[ind[i]][0] * w[i];
					pt[1] += vars[ind[i]][1] * w[i];
				}
				if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
					continue;
				Vector3d pixG = interpolation_util::bilinear<uchar, 3>(oriGrid.data, oriGrid.cols, oriGrid.rows, pt);
				Vector3d pixO = interpolation_util::bilinear<uchar, 3>(inputImg.data, inputImg.cols, inputImg.rows, pt);
				vis.at<Vec3b>(y, x) = Vec3b((uchar) pixG[0], (uchar) pixG[1], (uchar) pixG[2]);
				outputImg.at<Vec3b>(y, x) = Vec3b((uchar) pixO[0], (uchar) pixO[1], (uchar) pixO[2]);
			}
		}

		vector<Vector2d> resGrid(grid2.size());
		for(auto i=0; i<resGrid.size(); ++i){
			resGrid[i][0] = vars[i][0];
			resGrid[i][1] = vars[i][1];
		}
		vector<Vector2d> refPt3(refPt.size(), Vector2d(0,0));
		for(auto i=0; i<refPt.size(); ++i){
			Vector4i ind;
			Vector4d w;
			getGridIndAndWeight(refPt[i], ind, w);
			for(auto j=0; j<4; ++j)
				refPt3[i] += resGrid[ind[j]] * w[j];
		}

		Mat resGridImg;
		visualizeGrid(resGrid, resGridImg);
//		sprintf(buffer, "%s/temp/Grid%05d_2.jpg", file_io.getDirectory().c_str(), id);
//		imwrite(buffer, resGridImg);
//
		Mat resRefPtImg = resGridImg.clone();
		drawKeyPoints(resRefPtImg, refPt3);
		sprintf(buffer, "%s/temp/refPt%05d_2.jpg", file_io.getDirectory().c_str(), id);
		imwrite(buffer, resRefPtImg);
	}

	void drawKeyPoints(cv::Mat& img, const std::vector<Eigen::Vector2d>& pts){
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(128, 255);
		for (auto i = 0; i < pts.size(); ++i) {
			uchar ranR = (uchar) distribution(generator);
			uchar ranG = (uchar) distribution(generator);
			uchar ranB = (uchar) distribution(generator);
			cv::circle(img, cv::Point(pts[i][0], pts[i][1]), 3, cv::Scalar(ranR, ranG, ranB), 2);
			//printf("(%.2f,%.2f), (%.2f,%.2f)\n", refPt[i][0], refPt[i][1], srcPt[i][0], srcPt[i][1]);
		}
	}


} //namespace dynamic_stereo