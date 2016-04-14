//
// Created by Yan Hang on 4/9/16.
//

#include "dynamic_utility.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
	namespace utility {
		typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;
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

			vector <TriMesh::VertexHandle> vhandle((size_t) (w * h));
			Vector3d camcenter = cam.GetPosition();
			TriMesh::VertexHandle ch = mesh.add_vertex(TriMesh::Point(camcenter[0], camcenter[1], camcenter[2]));
			mesh.set_color(ch, TriMesh::Color(255, 0, 0));

			int vid = 0;
			for (auto y = 0; y < h; ++y) {
				for (auto x = 0; x < w; ++x) {
					if (depth(x, y) <= 0)
						continue;
					cv::Vec3b pix = image.at<Vec3b>(y, x);
					Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(x * downsample, y * downsample));
					//ray.normalize();
					Vector3d spt = cam.GetPosition() + ray * depth(x, y);
					vhandle[vid] = mesh.add_vertex(TriMesh::Point(spt[0], spt[1], spt[2]));
					mesh.set_color(vhandle[vid], TriMesh::Color(pix[2], pix[1], pix[0]));
					vid++;
				}
			}

			OpenMesh::IO::Options wopt;
			wopt += OpenMesh::IO::Options::VertexColor;
			CHECK(OpenMesh::IO::write_mesh(mesh, path, wopt)) << "Can not write ply file " << path;
		}

		void temporalMedianFilter(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, const int r){
			CHECK(!input.empty());
			const int h = input[0].rows;
			const int w = input[0].cols;
			output.resize(input.size(), cv::Mat(h,w,CV_8UC3,Scalar(1,1,1)));
			printf("Applying median filter, r = %d\n", r);
			for(auto i=0; i<input.size(); ++i) {
				int s, e;
				if (i - r < 0) {
					s = 0;
					e = 2 * r + 1;
				}
				else if (i + r >= input.size()) {
					s = (int)input.size() - 2 * r - 2;
					e = (int)input.size() - 1;
				} else {
					s = i - r;
					e = i + r;
				}
				for(auto y=0; y<h; ++y){
					for(auto x=0; x<w; ++x){
						vector<int> rc,gc,bc;
						for(auto t = s; t <= e; ++t){
							Vec3b pix = input[t].at<Vec3b>(y,x);
							rc.push_back(pix[0]);
							gc.push_back(pix[1]);
							bc.push_back(pix[2]);
						}
						nth_element(rc.begin(), rc.begin()+r, rc.end());
						nth_element(gc.begin(), gc.begin()+r, gc.end());
						nth_element(bc.begin(), bc.begin()+r, bc.end());
						output[i].at<Vec3b>(y,x) = Vec3b((uchar)rc[r], (uchar)gc[r], (uchar)bc[r]);
					}
				}
			}
		}

		void computeMinMaxDepth(const SfMModel& sfm, const int refId, double& min_depth, double& max_depth){
			const theia::Camera& cam = sfm.getCamera(refId);
			min_depth = -1;
			max_depth = -1;

			vector<theia::TrackId> trackIds = sfm.getView(refId)->TrackIds();
			printf("number of tracks:%lu\n", trackIds.size());
			vector<double> depths;
			for (const auto tid: trackIds) {
				const theia::Track *t = sfm.reconstruction.Track(tid);
				Vector4d spacePt = t->Point();
				Vector2d imgpt;
				double curdepth = cam.ProjectPoint(spacePt, &imgpt);
				if (curdepth > 0)
					depths.push_back(curdepth);
			}
			//ignore furthest 1% and nearest 1% points
			const double lowRatio = 0.01;
			const double highRatio = 0.99;
			const size_t lowKth = (size_t) (lowRatio * depths.size());
			const size_t highKth = (size_t) (highRatio * depths.size());
			//min_disp should be correspond to high depth
			nth_element(depths.begin(), depths.begin() + lowKth, depths.end());
			CHECK_GT(depths[lowKth], 0.0);
			min_depth = depths[lowKth];
			nth_element(depths.begin(), depths.begin() + highKth, depths.end());
			CHECK_GT(depths[highKth], 0.0);
			max_depth = depths[highKth];
		}

		void verifyEpipolarGeometry(const FileIO& file_io,
									const SfMModel& sfm,
									const int id1, const int id2,
									const Eigen::Vector2d& pt,
									cv::Mat &imgL, cv::Mat &imgR) {
			CHECK_GE(id1, 0);
			CHECK_GE(id2, 0);
			CHECK_LT(id1, file_io.getTotalNum());
			CHECK_LT(id2, file_io.getTotalNum());

			Mat tempM = imread(file_io.getImage(id1));
			const int width = tempM.cols;
			const int height = tempM.rows;
			CHECK_GE(pt[0], 0);
			CHECK_GE(pt[1], 0);
			CHECK_LT(pt[0], width);
			CHECK_LT(pt[1], height);

			const theia::Camera& cam1 = sfm.getCamera(id1);
			const theia::Camera& cam2 = sfm.getCamera(id2);

			Vector3d ray1 = cam1.PixelToUnitDepthRay(pt);
			//ray1.normalize();
			imgL = imread(file_io.getImage(id1));
			imgR = imread(file_io.getImage(id2));
			cv::circle(imgL, cv::Point(pt[0], pt[1]), 2, cv::Scalar(0,0,255), 2);

			double min_depth, max_depth;
			computeMinMaxDepth(sfm, id1, min_depth, max_depth);

			CHECK_GT(min_depth, 0.0);
			CHECK_GT(max_depth, 0.0);
			double min_disp = 1.0 / max_depth;
			double max_disp = 1.0 / min_depth;
			printf("min depth:%.3f, max depth:%.3f\n", min_depth, max_depth);

			double cindex = 0.0;
			double steps = 1000;
			for(double i=min_disp; i<=max_disp; i+=(max_disp-min_disp)/steps){
				Vector3d curpt = cam1.GetPosition() + ray1 * 1.0 / i;
				Vector2d imgpt;
				double depth = cam2.ProjectPoint(curpt.homogeneous(), &imgpt);
				if(depth < 0)
					continue;
				imgpt = imgpt;
				//printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
				cv::Point cvpt(((int)imgpt[0]), ((int)imgpt[1]));
				cv::circle(imgR, cvpt, 1, cv::Scalar(255 - cindex * 255.0, 0 ,cindex * 255.0));
				cindex += 1.0 / steps;
			}
		}

	}//namespace utility
}//namespace dynamic_stereo

