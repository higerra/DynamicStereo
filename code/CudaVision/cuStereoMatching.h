//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUSTEREOMATCHING_H
#define DYNAMICSTEREO_CUSTEREOMATCHING_H

#include <cstring>
#include <glog/logging.h>
#include <vector>
#include "cuCamera.h"
#include "cuStereoUtil.h"

namespace CudaVision{

    static const int BLOCKDIM = 2;
    static const int MAXFRAME = 60;
    static const int MAXPATCHSIZE = 50;

    __constant__ CudaCamera<double> device_cameras[MAXFRAME];
    __constant__ CudaCamera<double> device_refCam;

    //--------------------------------------
    //Kernel
    //--------------------------------------
    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const unsigned char* refImage, const int width, const int height, const int N,
                                         const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam, const TCam* spacePt, const int resolution, const int R, TOut* output);




    template <typename TCam, typename TOut, int WIDTH, int HEIGHT>
    class CudaStereoMatching {
    public:
        CudaStereoMatching(const int N_, const int resolution_, const int R_)
                : width(WIDTH), height(HEIGHT), blockSize(WIDTH, HEIGHT), N(N_), resolution(resolution_), R(R_),
                  images(nullptr), refImage(nullptr), spacePt(nullptr), output(nullptr) {
            allocate();
        }

        ~CudaStereoMatching(){
            HandleCuError(cudaFree(images));
            HandleCuError(cudaFree(refImage));
            HandleCuError(cudaFree(spacePt));
            HandleCuError(cudaFree(output));
        }
        void run(const unsigned char* host_images, const unsigned char* host_refImage,
                 const CudaCamera<TCam>* host_cameras, const CudaCamera<TCam>* host_refCam,
                 const TCam* host_spts,
                 std::vector<TOut>& result);

    private:
        const int width;
        const int height;
        const dim3 blockSize;

        const int N;
        const int resolution;
        const int R;


        //pointer to device data
        unsigned char *images;
        unsigned char *refImage;
        TCam *spacePt;
        TOut *output;

        void allocate(){
            LOG(INFO) << "Allocate device memory";
            LOG(INFO) << "Allocate images";
            CudaVision::HandleCuError(cudaMalloc((void**)& images, width * height * N * 3 * sizeof(unsigned char)));
            LOG(INFO) << "Allocate refImage";
            CudaVision::HandleCuError(cudaMalloc((void**)& refImage, width * height * 3 * sizeof(unsigned char)));
            LOG(INFO) << "Allocate space points";
            CudaVision::HandleCuError(cudaMalloc((void**)& spacePt, width * height * resolution * 3 * sizeof(TCam)));
            LOG(INFO) << "Allocate output";
            CudaVision::HandleCuError(cudaMalloc((void**)& output, width * height * resolution * sizeof(TOut)));

        }
    };



    ///////////////////////////////////////////////////////////////////
    //Implementation
    ///////////////////////////////////////////////////////////////////

    template<typename TCam, typename TOut, int WIDTH, int HEIGHT>
    void CudaStereoMatching<TCam, TOut, WIDTH, HEIGHT>::run(const unsigned char* host_images, const unsigned char* host_refImage,
                                                            const CudaCamera<TCam>* host_cameras, const CudaCamera<TCam>* host_refCam,
                                                            const TCam* host_spts,
                                                            std::vector<TOut>& result) {
        if(result.size() != width * height * resolution)
            result.resize(width * height * resolution);
        HandleCuError(cudaMemcpyToSymbol(device_cameras, host_cameras, N * sizeof(CudaVision::CudaCamera<TCam>)));
        HandleCuError(cudaMemcpyToSymbol(&device_refCam, host_refCam, sizeof(CudaVision::CudaCamera<TCam>)));

        LOG(INFO) << "Uploading images";
        CudaVision::HandleCuError(cudaMemcpy(images, host_images, width * height * N * 3 * sizeof(unsigned char),
                                             cudaMemcpyHostToDevice));
        CudaVision::HandleCuError(cudaMemcpy(refImage, host_refImage, width * height * 3 * sizeof(unsigned char),
                                             cudaMemcpyHostToDevice));
        LOG(INFO) << "Uploading space points";
        CudaVision::HandleCuError(
                cudaMemcpy(spacePt, host_spts, width * height * resolution * 3 * sizeof(TCam), cudaMemcpyHostToDevice));

        //call kernel
        stereoMatchingKernel<TCam, TOut> <<<blockSize, BLOCKDIM>>>(images, refImage, width, height, N, device_cameras, &device_refCam, spacePt, resolution, R, output);

        LOG(INFO) << "Copy back result";
        CudaVision::HandleCuError(cudaMemcpy(result.data(), output, width * height * resolution * sizeof(TOut), cudaMemcpyDeviceToHost));
    }



    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const unsigned char* refImage, const int width, const int height, const int N,
                                         const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam, const TCam* spacePt, const int resolution, const int R, TOut* output){
        int x = blockIdx.x;
        int y = blockIdx.y;

        __shared__ TOut refPatch[MAXPATCHSIZE * 3];

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

        //allocate memory
        TOut nccArray[MAXFRAME];
        TOut newPatch[MAXPATCHSIZE * 3];

        const int patchSize = (2*R+1) * (2*R+1);
        //the number of threads in each block can be fewer than resolution
        for(int d=threadIdx.x; d < resolution; d += blockDim.x){
            //position inside output array
            int outputOffset = (y * width + x) * resolution + d;

            for(int v=0; v<N; ++v) {
                //reset new patch
                for(auto i=0; i<MAXPATCHSIZE * 3; ++i)
                    newPatch[i] = -1;
                //project space point and extract pixel
                int ind = 0;
                for (int dx = -1 * R; dx <= R; ++dx) {
                    for (int dy = -1 * R; dy <= R; ++dy) {
                        int curx = x + dx, cury = y + dy;
                        if (curx >= 0 && curx < width && cury >= 0 && cury < height) {
                            TCam projected[2];
                            cameras[v].projectPoint(spacePt + ((cury * width + curx) * resolution + d) * 3,
                                                    projected);
                            if (projected[0] >= 0 && projected[1] >= 0 && projected[0] < width - 1 &&
                                projected[1] < height - 1) {
                                bilinearInterpolation<unsigned char, TCam, TOut>(images + width * height * v * 3, width,
                                                                                 projected, newPatch + ind * 3);
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
        }
    }
}//namespace CudaVision
#endif //DYNAMICSTEREO_CUSTEREOMATCHING_H
