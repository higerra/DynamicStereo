//
// Created by yanhang on 5/24/16.
//

#include "regiondescriptor.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
	namespace Feature{
		Region::Region(const std::vector<cv::Point> &locs_, const cv::Mat &img_):
				locs(locs_), img(img_), kBin(8){
			CHECK(img.data);
			width = (float)img.cols;
			height = (float)img.rows;

			pixs.reserve(locs.size());
			for(const auto& pt: locs){
				CHECK_GE(pt.x, 0);
				CHECK_GE(pt.y, 0);
				CHECK_LT(pt.x, img.cols);
				CHECK_LT(pt.y, img.rows);
				Vec3b curpix = img.at<Vec3b>(pt.y, pt.x);
				pixs.push_back(Vector3f(curpix[0], curpix[1], curpix[2]));
			}
		}

		void Region::computeFeatures() {
			CHECK(img.data);
			CHECK(!locs.empty());
			CHECK_EQ(locs.size(), pixs.size());
			//compute features of a region
			//area
			area = (int)locs.size();
			//convex hull
			vector<cv::Point> chull;
			cv::convexHull(locs, chull);
			area_chall = (float)cv::contourArea(chull);
			
			//bounding box
			cv::RotatedRect box = cv::minAreaRect(locs);

		}

	}//namespace Fature
}//namespace dynamic_stereo