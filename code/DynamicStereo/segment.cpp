//
// Created by yanhang on 4/9/16.
//

#include "dynamicstereo.h"
#include "external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    //binaryMask: in ORIGINAL resolution!
    void DynamicStereo::dynamicSegment(const Depth &disparity, cv::Mat& binaryMask) const {
        CHECK_EQ(disparity.getWidth(), model->width);
        CHECK_EQ(disparity.getHeight(), model->height);
        char buffer[1024] = {};

	    vector<Mat> warpped;
	    warpToAnchor(disparity, segMask, 0, (int)images.size() - 1, warpped);
	    for(auto i=0; i<warpped.size(); ++i){
		    sprintf(buffer, "%s/temp/segwarp_b%05d_f%05d.jpg", file_io.getDirectory().c_str(), anchor, i+offset);
		    imwrite(buffer, warpped[i]);
	    }




//        {
//            //debug: visualize matching cost
//            Depth mCost(model->width, model->height, 0.0);
//            for(auto y=0; y<model->height; ++y){
//                for(auto x=0; x<model->width; ++x){
//                    CHECK_LT(disparity(x,y), model->nLabel);
//                    if(segMask.at<uchar>(y*downsample, x*downsample) < 200)
//                        mCost(x,y) = 0.0;
//                    else
//                        mCost(x,y) = model->operator()(y*width+x, disparity(x,y));
//                }
//            }
//            sprintf(buffer, "%s/temp/segConf%05d.jpg", file_io.getDirectory().c_str(), anchor);
//            mCost.saveImage(string(buffer), 0.25);
//        }
    }
}//namespace dynamic_stereo
