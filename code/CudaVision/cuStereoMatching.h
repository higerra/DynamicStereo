//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUSTEREOMATCHING_H
#define DYNAMICSTEREO_CUSTEREOMATCHING_H

#include <cstring>
#include "cuCamera.h"
#include "cuStereoUtil.h"

namespace CudaVision{
    //--------------------------------------
    //Kernel
    //--------------------------------------

    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const unsigned char* refImage,
                                         const int width, const int height, const int N,
                                         const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam,
                                         const TCam min_disp, const TCam max_disp,
                                         const TCam* spacePt, const int resolution, const int R, const int offset,
                                         TOut* output){
        int x = blockIdx.x;
        int y = blockIdx.y;
        int d = threadIdx.x + offset;

        const int patchSize = (2*R+1) * (2*R+1);
        __shared__ TOut refPatch[50];

        //the first thread in the block create reference patch
        if(threadIdx.x == 0){
            int ind = 0;
            for (int dx = -1 * R; dx <= R; ++dx) {
                for (int dy = -1 * R; dy <= R; ++dy) {
                    int curx = x + dx, cury = y + dy;
                    if (curx >= 0 && curx < width && cury >= 0 && cury < height) {
                        refPatch[3 * ind] = refImage[3 * (cury * width + curx)];
                        refPatch[3 * ind + 1] = refImage[3 * (cury * width + curx) + 1];
                        refPatch[3 * ind + 2] = refImage[3 * (cury * width + curx) + 2];
                    }else{
                        refPatch[3 * ind] = -1;
                        refPatch[3 * ind + 1] = -1;
                        refPatch[3 * ind + 2] = -1;
                    }
                    ind++;
                }
            }
        }

        __syncthreads();

        int outputOffset = (y * width + x) * blockDim.x + d - offset;

        TCam depth = 1.0/(min_disp + d * (max_disp - min_disp) / (TCam) resolution);

        //allocate memory
        TOut* nccArray = new TOut[N];
        TOut* newPatch = new TOut[(2*R+1) * (2*R+1) * 3];
        memset(newPatch, -1, (2*R+1) * (2*R+1) *3 * sizeof(TOut));

        for(int v=0; v<N; ++v) {
            //project space point and extract pixel
            int ind = 0;
            for (int dx = -1 * R; dx <= R; ++dx) {
                for (int dy = -1 * R; dy <= R; ++dy) {
                    int curx = x + dx, cury = y + dy;
                    if (curx >= 0 && curx < width && cury >= 0 && cury < height) {
                        TCam projected[2];
                        cameras[v].projectPoint(spacePt + (cury * width + curx) * 3, projected[2]);
                        if(projected[0] >= 0 && projected[1] >= 0 && projected[0] < width - 1 && projected[1] < height - 1){
                            bilinearInterpolation<unsigned char, TCam, TOut>(images[v], projected, newPatch + ind * 3);
                        }
                    }
                    ind++;
                }
            }

            //compute NCC
            TOut mean1 = 0, mean2 = 0, count = 0;
            for(int i=0; i<patchSize; ++i){
                if(newPatch[3*i] >= 0 && refPatch[3*i] >= 0){
                    mean1 += refPatch[3*i] + refPatch[3*i+1] + refPatch[3*i+2];
                    mean2 += newPatch[3*i] + newPatch[3*i+1] + newPatch[3*i+2];
                    count += 1;
                }
            }
            mean1 /= (3 * count);
            mean2 /= (3 * count);

            TOut var1 = 0, var2 = 0;
            for(int i=0; i<patchSize; ++i){
                if(newPatch[3*i] >= 0 && refPatch[3*i] >= 0){
                    var1 += (refPatch[3*i] - mean1) * (refPatch[3*i] - mean1) +
                            (refPatch[3*i + 1] - mean1) * (refPatch[3*i + 1] - mean1) +
                            (refPatch[3*i + 2] - mean1) * (refPatch[3*i + 2] - mean1);
                    var2 += (newPatch[3*i] - mean2) * (newPatch[3*i] - mean2) +
                            (newPatch[3*i + 1] - mean2) * (newPatch[3*i + 1] - mean2) +
                            (newPatch[3*i + 2] - mean2) * (newPatch[3*i + 2] - mean2);
                }
            }
            if(var1 < FLT_EPSILON || var2 < FLT_EPSILON)
                nccArray[v] = 0;
            else {
                var1 = sqrt(var1 / count);
                var2 = sqrt(var2 / count);
                TOut ncc = 0;
                for (int i = 0; i < patchSize; ++i) {
                    if (newPatch[3 * i] >= 0 && refPatch[3 * i] >= 0) {
                        ncc += (refPatch[3 * i] - mean1) * (newPatch[3 * i] - mean2) +
                               (refPatch[3 * i + 1] - mean1) * (newPatch[3 * i + 1] - mean2) +
                               (refPatch[3 * i + 2] - mean1) * (newPatch[3 * i + 2] - mean2);
                    }
                }
                nccArray[v] = ncc / (var1 * var2 * (N-1));
            }
        }

        output[outputOffset] = find_nth(nccArray, N, N/2);
        delete nccArray;
        delete newPatch;
    };

}//namespace CudaVision
#endif //DYNAMICSTEREO_CUSTEREOMATCHING_H
