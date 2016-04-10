//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"

namespace dynamic_stereo{

    DynamicSegment::DynamicSegment(const FileIO &file_io_, const int anchor_, const int downsample_,
    const std::vector<Depth>& depths_, const std::vector<int>& depthInd_):
            file_io(file_io_), anchor(anchor_), downsample(downsample_), depths(depths_), depthInd(depthInd_) {
        sfmModel.init(file_io.getReconstruction());

    }

}//namespace dynamic_stereo
