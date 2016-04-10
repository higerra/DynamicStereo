//
// Created by Yan Hang on 4/9/16.
//

#include "dynamicstereo.h"
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
				for (auto x = 0; x < w; ++x, ++vid) {
					if (depth(x, y) <= 0)
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

	}//namespace utility
}//namespace dynamic_stereo


