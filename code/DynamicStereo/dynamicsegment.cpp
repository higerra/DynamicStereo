//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    DynamicSegment::DynamicSegment(const FileIO &file_io_, const int anchor_, const int tWindow_, const int downsample_,
    const std::vector<Depth>& depths_, const std::vector<int>& depthInd_):
            file_io(file_io_), anchor(anchor_), downsample(downsample_), depths(depths_), depthInd(depthInd_) {
        sfmModel.init(file_io.getReconstruction());

	    //load image
	    if (anchor - tWindow_ / 2 < 0) {
		    offset = 0;
		    CHECK_LT(offset + tWindow_, file_io.getTotalNum());
	    } else if (anchor + tWindow_ / 2 >= file_io.getTotalNum()) {
		    offset = file_io.getTotalNum() - 1 - tWindow_;
		    CHECK_GE(offset, 0);
	    } else
		    offset = anchor - tWindow_ / 2;

	    images.resize((size_t) tWindow_);
	    for (auto i = 0; i < images.size(); ++i) {
		    images[i] = imread(file_io.getImage(i + offset));
		    for (auto y = 0; y < images[i].rows; ++y) {
			    for (auto x = 0; x < images[i].cols; ++x) {
				    if (images[i].at<Vec3b>(y, x) == Vec3b(0, 0, 0))
					    images[i].at<Vec3b>(y, x) = Vec3b(1, 1, 1);
			    }
		    }
	    }
	    CHECK(!images.empty());
	    width = images[0].cols;
	    height = images[0].rows;

	    CHECK_EQ(depths.size(), depthInd.size());
	    for(auto i=0; i<depthInd.size(); ++i){
			depths[i].updateStatics();
		    if(depthInd[i] == anchor){
			    refDepth = depths[i];
		    }
	    }

	    CHECK_EQ(refDepth.getWidth(), width / downsample);
	    CHECK_EQ(refDepth.getHeight(), height / downsample);
    }

//	void DynamicSegment::getGeometryConfidence(Depth &geoConf) const {
//		geoConf.initialize(width, height, 0.0);
//		const theia::Camera& refCam = sfmModel.getCamera(anchor);
//		const double large = 1000000;
//
//		for(auto y=downsample; y<height-downsample; ++y){
//			for(auto x=downsample; x<width-downsample; ++x){
//				Vector2d refPt((double)x/(double)downsample, (double)y/(double)downsample);
//				Vector3d spt = refCam.GetPosition() + refCam.PixelToUnitDepthRay(Vector2d(x,y)) * refDepth.getDepthAt(refPt);
//				vector<double> repoError;
//
//				for(auto j=0; j<depthInd.size(); ++j){
//					Vector2d imgpt;
//					const theia::Camera& cam2 = sfmModel.getCamera(depthInd[j]);
//					double d = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
//					imgpt /= (double)downsample;
//					if(d > 0 && imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < depths[j].getWidth()-1 && imgpt[1] < depths[j].getHeight()-1){
//						double depth2 = depths[j].getDepthAt(imgpt);
//						double zMargin = depths[j].getMedianDepth() / 5;
//						if(d <= depth2 + zMargin) {
//							Vector3d spt2 = cam2.PixelToUnitDepthRay(imgpt * downsample) * depth2 + cam2.GetPosition();
//							Vector2d repoPt;
//							double repoDepth = refCam.ProjectPoint(spt2.homogeneous(), &repoPt);
//							double dis = (repoPt - Vector2d(x,y)).norm();
//							repoError.push_back(dis);
//						}
//					}
//				}
//
//				//take average
//				if(!repoError.empty())
//					geoConf(x,y) = std::accumulate(repoError.begin(), repoError.end(), 0.0) / (double)repoError.size();
//			}
//		}
//	}


	void DynamicSegment::segment(const std::vector<cv::Mat> &warppedImg, cv::Mat &result) const {
		result = Mat(height, width, CV_8UC1, Scalar(255));


	}

}//namespace dynamic_stereo
