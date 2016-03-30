//
// Created by yanhang on 3/28/16.
//

#include "gridwarpping.h"
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

	void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
	                                       Eigen::Vector4d &w) const {
		CHECK_LE(pt[0], width-1);
		CHECK_LE(pt[1], height-1);
		int x = (int) floor(pt[0] / blockW);
		int y = (int) floor(pt[1] / blockH);

		//////////////
		// 1--2
		// |  |
		// 4--3
		/////////////
		ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
		               (y + 1) * (gridW + 1) + x);
		for (auto i = 0; i < 4; ++i) {
			CHECK_LT(ind[i], gridLoc.size());
			w[i] = (pt - gridLoc[ind[i]]).norm();
		}
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


	struct WarpFunctorData{
	public:
		WarpFunctorData(const Vector2d& tgt_, const Vector4d& w_): tgt(tgt_), w(w_){}
		template<typename T>
		bool operator()(const T* const g1, const T* const g2, const T* const g3, const T* const g4, T* residual) const{
			T x = g1[0] * w[0] + g2[0] * w[1] + g3[0] * w[2] + g4[0] * w[3];
			T y = g1[1] * w[0] + g2[1] * w[1] + g3[1] * w[2] + g4[1] * w[3];
			T diffx = x - (T)tgt[0];
			T diffy = y - (T)tgt[1];
			residual[0] = ceres::sqrt(diffx * diffx + diffy * diffy);
			return true;
		}
	private:
		const Vector2d tgt;
		const Vector4d w;
	};

	void GridWarpping::computeWarppingField(const std::vector<Eigen::Vector2d> &refPt,
	                                        const std::vector<Eigen::Vector2d> &srcPt,
	                                        std::vector<std::vector<Eigen::Vector2d> > &wf, cv::Mat &vis) const {
		CHECK_EQ(refPt.size(), srcPt.size());
		vector<vector<double> > vars((size_t)(gridW+1) * (gridH+1));
		for(auto& v: vars)
			v.resize(2);
		for(auto i=0; i<gridLoc.size(); ++i){
			vars[i][0] = gridLoc[i][0];
			vars[i][1] = gridLoc[i][1];
		}

		ceres::Problem problem;
		for(auto i=0; i<refPt.size(); ++i) {
			const double &xd = srcPt[i][0];
			const double &yd = srcPt[i][1];
			int x = (int) floor(xd / blockW);
			int y = (int) floor(yd / blockH);
			Vector4i ind(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
			             (y + 1) * (gridW + 1) + x);
			Vector4d bw((xd - x * blockW) * (yd - y * blockH),
			            ((x + 1) * blockW - xd) * (yd - y * blockH),
			            ((x + 1) * blockW - xd) * ((y + 1) * blockH - yd),
			            (xd - x * blockW) * ((y + 1) * blockH - yd));
			double bwsum = bw[0] + bw[1] + bw[2] + bw[3];
			CHECK_GT(bwsum, 0.0);
			bw = bw / bwsum;
			problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WarpFunctorData,1,2,2,2,2>(new WarpFunctorData(refPt[i], bw)), NULL,
			                         vars[ind[0]].data(), vars[ind[1]].data(), vars[ind[2]].data(), vars[ind[3]].data());

		}

		ceres::Solver::Options options;
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		wf.resize(width);
		for(auto& w: wf)
			w.resize(height);

	}
} //namespace dynamic_stereo