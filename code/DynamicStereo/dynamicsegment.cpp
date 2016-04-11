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
		    if(depthInd[i] == anchor){
			    refDepth = depths[i];
			    break;
		    }
	    }

	    CHECK_EQ(refDepth.getWidth(), width / downsample);
	    CHECK_EQ(refDepth.getHeight(), height / downsample);


    }

	void DynamicSegment::getGeometryConfidence(Depth &geoConf) const {
		geoConf.initialize(width, height, 0.0);
		const theia::Camera& refCam = sfmModel.getCamera(anchor);
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vector3d spt = refCam.GetPosition() + refCam.PixelToUnitDepthRay(Vector2d(x,y)) * refDepth(x/downsample, y/downsample);
			}
		}
	}

}//namespace dynamic_stereo
