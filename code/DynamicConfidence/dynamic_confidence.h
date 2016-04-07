//
// Created by yanhang on 4/7/16.
//

#ifndef DYNAMICSTEREO_DYNAMIC_CONFIDENCE_H
#define DYNAMICSTEREO_DYNAMIC_CONFIDENCE_H
#include <iostream>
#include <theia/theia.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <glog/logging.h>

#include "../base/file_io.h"
#include "../base/opticalflow.h"
#include "../base/depth.h"
#include "../base/utility.h"

namespace dynamic_stereo {
    class DynamicConfidence {
    public:
        DynamicConfidence(const FileIO& file_io_):
                file_io(file_io_), max_tWindow(100){
            init();
        }
        void init();
        void run(const int anchor, Depth& confidence);
    private:
        const FileIO &file_io;
        typedef std::pair<int, theia::ViewId> IdPair;
        std::vector<IdPair> orderedId;
        theia::Reconstruction reconstruction;

        const int max_tWindow;
    };

}
#endif //DYNAMICSTEREO_DYNAMIC_CONFIDENCE_H
