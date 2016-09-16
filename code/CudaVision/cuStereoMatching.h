//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUSTEREOMATCHING_H
#define DYNAMICSTEREO_CUSTEREOMATCHING_H

#include "cuCamera.h"
#include "cuStereoUtil.h"

namespace CudaVision{
    //--------------------------------------
    //Kernel
    //--------------------------------------

    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const int width, const int height, const int N,
                                         const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam,
                                         const TCam min_disp, const TCam max_disp,
                                         const TCam* rays, const int resolution, const int R, const int offset,
                                         TOut* output){
        int x = blockIdx.x;
        int y = blockIdx.y;
        int d = threadIdx.x + offset;

        int outputOffset = (y * width + x) * blockDim.x + d - offset;

        TCam depth = 1.0/(min_disp + d * (max_disp - min_disp) / (TCam) resolution);

        //allocate memory
        TOut* nccArray = new TOut[N - 1];

        //compute matching score in place to save some memory
        for(int v=0; v<N; ++v) {
            //compute space point of the pixel patch
            TCam spt[4] = {};
            for (int dx = -1 * R; dx <= R; ++dx) {
                for (int dy = -1 * R; dy <= R; ++dy) {
                    int curx = x + dx, cury = y + dy;
                    if (curx >= 0 && curx < width && cury >= 0 && cury < height) {
                        for (int i = 0; i < 4; ++i) {
                            spt[i] = refCam->getPosition()[i] + depth * rays[(cury*width+curx) * 3 + i];
                        }
                    }
                }
            }
        }

        output[outputOffset] = find_nth(nccArray, N-1, (N-1)/2);
        delete nccArray;
    };

}//namespace CudaVision
#endif //DYNAMICSTEREO_CUSTEREOMATCHING_H
