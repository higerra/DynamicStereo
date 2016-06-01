//
// Created by yanhang on 5/24/16.
//

#include "regiondescriptor.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
	namespace Feature{
		void computeFeatures(const std::vector<cv::Mat>& images,
									 const std::vector<cv::Point>& locs,
									 std::vector<double>& feature) {
			double area;
			int kEdge;
			double aspectRatio;
			double area_chall;
			double area_bbox;
			std::vector<float> hist_diff;
			std::vector<float> hist_color;

			const int kBin = 8;

			//sanity check
			CHECK(!images.size());
			double width = (double)images[0].cols;
			double height = (double)images[0].rows;
			CHECK(!locs.empty());
			for(const auto& loc: locs){
				CHECK_GE(loc.x, 0);
				CHECK_GE(loc.y, 0);
				CHECK_LT(loc.x, width);
				CHECK_LT(loc.y, height);
			}

			//compute features of a region
			//area
			area = (double)locs.size() / width / height;
			//convex hull
			vector<cv::Point> chull;
			cv::convexHull(locs, chull);
			area_chall = (float)cv::contourArea(chull);
			
			//bounding box: aspect ratio and area
			cv::RotatedRect box = cv::minAreaRect(locs);
			vector<cv::Point2f> boxPt(4);
			box.points(boxPt.data());
			double len1 = cv::norm(boxPt[1] - boxPt[0]);
			double len2 = cv::norm(boxPt[2] - boxPt[1]);
			CHECK_GT(len1, 0);
			CHECK_GT(len2, 0);
			if(len1 >= len2)
				aspectRatio = len1 / len2;
			else
				aspectRatio = len2 / len1;
			area_bbox = len1 * len2 / width / height;

			//color related
			//histogram
			hist_diff.resize((size_t)kBin * 3, 0.0);
			hist_color.resize((size_t)kBin * 3, 0.0);
			const float binUnitDiff = 512 / (float)kBin;
			const float binUnitColor = 256 / (float)kBin;
			const int stride = (int)images.size() / 2;
			for(const auto& loc: locs){
				//color
				for(auto v=0; v<images.size(); ++v){
					Vec3b pixb = images[v].at<Vec3b>(loc);
					for(auto c=0; c<3; ++c){
						int bid = (int)floor((float)pixb[c] / binUnitColor);
						CHECK_LT(kBin*c+bid, hist_color.size());
						hist_color[kBin*c+bid] += 1.0;
					}
				}
				//color change
				for(auto v=0; v<images.size()-stride; ++v){
					Vec3b pix1 = images[v].at<Vec3b>(loc);
					Vec3b pix2 = images[v+stride].at<Vec3b>(loc);
					Vector3f pix1f((float)pix1[0], (float)pix1[1], (float)pix1[2]);
					Vector3f pix2f((float)pix2[0], (float)pix2[1], (float)pix2[2]);
					for(auto c=0; c<3; ++c){
						int bid = (int)floor((pix2f[c] - pix1f[c] + 256) / binUnitDiff);
						CHECK_LT(kBin*c+bid, hist_diff.size());
						hist_diff[bid] += 1.0;
					}
				}
			}
			const double cut_thres = 0.1;
			normalizel2(hist_diff);
			normalizel2(hist_color);
			for(auto& v: hist_diff){
				if(v < cut_thres)
					v = 0;
			}
			for(auto& v: hist_color){
				if(v < cut_thres)
					v = 0;
			}
			normalizel2(hist_diff);
			normalizel2(hist_color);

			//TODO:line segments


			//compose feature vector
			feature.resize((size_t) kBin*6 + 4);
			const int histOffset = hist_diff.size() + hist_color.size();
			for(auto i=0; i<hist_diff.size(); ++i)
				feature[i] = (float)hist_diff[i];
			for(auto i=0; i<hist_color.size(); ++i)
				feature[i+kBin*3] = hist_color[i];
			feature[histOffset] = (float)area;
			feature[histOffset+1] = (float)area / area_chall;
			feature[histOffset+2] = (float)area / area_bbox;
			feature[histOffset+3] = aspectRatio;
		}

	}//namespace Fature
}//namespace dynamic_stereo