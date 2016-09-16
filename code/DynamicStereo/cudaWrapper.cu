//
// Created by yanhang on 9/16/16.
//

#include "cudaWrapper.h"
#include <vector>

namespace dynamic_stereo{
    void cudaCaller(const unsigned char* images, const unsigned char* refImage,
                    const int width, const int height, const int N,
                    const TCam* position, const TCam* axis,
                    const TCam min_disp, const TCam max_disp, const int resolution,
                    TOut* result){
        std::vector<CudaVision::CudaCamera<TCam> > cameras((size_t) N);
        
    }

}//namespace dynamic_stereo

