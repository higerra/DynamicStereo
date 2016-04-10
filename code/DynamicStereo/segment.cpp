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

	    {
		    //debug warpping
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
