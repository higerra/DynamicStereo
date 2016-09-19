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

    static const int BLOCKDIM = 128;
    static const int MAXFRAME = 50;
    static const int MAXPATCHSIZE = 25;

    __constant__ CudaCamera<double> device_cameras[MAXFRAME];
    __constant__ CudaCamera<double> device_refCam;

    //--------------------------------------
    //Kernel
    //--------------------------------------
    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const unsigned char* refImage,
                                         const int width, const int height, const int N,
                                         const TCam* rays,
                                         const TCam min_disp, const TCam max_disp, const TCam downsample, const int resolution, const int R, TOut* output);


    template<typename TCam, typename TOut>
    TOut stereoMatchingKernelDebug(const unsigned char* images, const unsigned char* refImage, const int width, const int height, const int N,
                                   const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam,
                                   const TCam* rays, const TCam min_disp, const TCam max_disp, const TCam downsample, const int resolution, const int R,
                                   const int x, const int y, const int d);

    template <typename TCam, typename TOut, int WIDTH, int HEIGHT>
    class CudaStereoMatching {
    public:
        CudaStereoMatching(const int N_, const int resolution_, const int R_, const double mind, const double maxd, const double downsample_)
                : width(WIDTH), height(HEIGHT), blockSize(WIDTH, HEIGHT), N(N_), resolution(resolution_), R(R_),
                  min_disp(mind), max_disp(maxd), downsample(downsample_),
                  images(nullptr), refImage(nullptr), rays(nullptr), output(nullptr) {
            allocate();
        }

        ~CudaStereoMatching(){
            HandleCuError(cudaFree(images));
            HandleCuError(cudaFree(refImage));
            HandleCuError(cudaFree(rays));
            HandleCuError(cudaFree(output));
            cudaDeviceSynchronize();
        }
        void run(const unsigned char* host_images, const unsigned char* host_refImage,
                 const CudaCamera<TCam>* host_cameras, const CudaCamera<TCam>* host_refCam,
                 const TCam* host_rays,
                 std::vector<TOut>& result);

    private:
        const int width;
        const int height;
        const dim3 blockSize;

        const int N;
        const int resolution;
        const int R;
        const TCam min_disp;
        const TCam max_disp;
        const TCam downsample;

        //pointer to device data
        unsigned char *images;
        unsigned char *refImage;
        TCam *rays;
        TOut *output;

        void allocate(){
            LOG(INFO) << "Allocate device memory";
            LOG(INFO) << "Allocate images";
            CudaVision::HandleCuError(cudaMalloc((void**)& images, width * height * N * 3 * sizeof(unsigned char)));
            LOG(INFO) << "Allocate refImage";
            CudaVision::HandleCuError(cudaMalloc((void**)& refImage, width * height * 3 * sizeof(unsigned char)));
            LOG(INFO) << "Allocate rays";
            CudaVision::HandleCuError(cudaMalloc((void**)& rays, width * height * 3 * sizeof(TCam)));
            LOG(INFO) << "Allocate output";
            CudaVision::HandleCuError(cudaMalloc((void**)& output, width * height * resolution * sizeof(TOut)));
            cudaDeviceSynchronize();
        }
    };



    ///////////////////////////////////////////////////////////////////
    //Implementation
    ///////////////////////////////////////////////////////////////////

    template<typename TCam, typename TOut, int WIDTH, int HEIGHT>
    void CudaStereoMatching<TCam, TOut, WIDTH, HEIGHT>::run(const unsigned char* host_images, const unsigned char* host_refImage,
                                                            const CudaCamera<TCam>* host_cameras, const CudaCamera<TCam>* host_refCam,
                                                            const TCam* host_rays,
                                                            std::vector<TOut>& result) {
        if(result.size() != width * height * resolution)
            result.resize(width * height * resolution);
        for(auto i=0; i<result.size(); ++i)
            result[i] = 0;

//        LOG(INFO) << "Uploading Cameras";
//        HandleCuError(cudaMemcpyToSymbol(device_cameras, host_cameras, N * sizeof(CudaVision::CudaCamera<TCam>)));
//        HandleCuError(cudaMemcpyToSymbol(device_refCam, host_refCam, sizeof(CudaVision::CudaCamera<TCam>)));
//        cudaDeviceSynchronize();
//
//        LOG(INFO) << "Uploading images";
//        CudaVision::HandleCuError(cudaMemcpy(images, host_images, width * height * N * 3 * sizeof(unsigned char),
//                                             cudaMemcpyHostToDevice));
//        CudaVision::HandleCuError(cudaMemcpy(refImage, host_refImage, width * height * 3 * sizeof(unsigned char),
//                                             cudaMemcpyHostToDevice));
//        cudaDeviceSynchronize();
//
//        LOG(INFO) << "Uploading rays";
//        CudaVision::HandleCuError(
//                cudaMemcpy(rays, host_rays, width * height * 3 * sizeof(TCam), cudaMemcpyHostToDevice));
//        cudaDeviceSynchronize();
//
//        //call kernel
//        LOG(INFO) << "Computing...";
//        stereoMatchingKernel<TCam, TOut> <<<blockSize, BLOCKDIM>>>(images, refImage, width, height, N, rays, min_disp, max_disp, downsample, resolution, R, output);
//        cudaDeviceSynchronize();
//
//        LOG(INFO) << "Copy back result";
//        CudaVision::HandleCuError(cudaMemcpy(result.data(), output, width * height * resolution * sizeof(TOut), cudaMemcpyDeviceToHost));
//        cudaDeviceSynchronize();
//
//        TOut saniv = 0;
//        for(int i=0; i<result.size(); ++i){
//            CHECK_GE(result[i], 0);
//            CHECK_LE(result[i], 1);
//            saniv += result[i];
//        }
//        LOG(INFO) << "Sum of result: " << saniv;
//        CHECK_GT(saniv, 0) << "Error copying data from GPU to CPU";

        //debug run
        const int dx = 181 / 2, dy = 1028 / 2, dd = 82;
        TOut debRes = stereoMatchingKernelDebug<TCam, TOut>(host_images, host_refImage, width, height, N, host_cameras, host_refCam, host_rays, min_disp, max_disp, downsample, resolution, R, dx, dy, dd);
        printf("Matching cost for (%d,%d) at %d is: %.3f\n", dx, dy, dd, debRes);
    }

    template<typename TCam, typename TOut>
    __global__ void stereoMatchingKernel(const unsigned char* images, const unsigned char* refImage, const int width, const int height, const int N,
                                         const TCam* rays, const TCam min_disp, const TCam max_disp, const TCam downsample, const int resolution, const int R, TOut* output){
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
        TOut nccArray[MAXFRAME] = {-1};
        TOut nccValid[MAXFRAME] = {};

        TOut newPatch[MAXPATCHSIZE * 3];

        const int patchSize = (2*R+1) * (2*R+1);
        //the number of threads in each block can be fewer than resolution
        for(int d=threadIdx.x; d < resolution; d += blockDim.x){
            //position inside output array
            int outputOffset = (y * width + x) * resolution + d;
            for(auto i=0; i<MAXFRAME; ++i)
                nccArray[i]  = -1;

            TCam depth = 1.0/(min_disp + d * (max_disp - min_disp) / (TCam) resolution);

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
                            //project points. Be careful with the downsample factor
                            TCam spt[4];
                            spt[0] = device_refCam.extrinsic[0 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3] * depth;
                            spt[1] = device_refCam.extrinsic[1 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3 + 1] * depth;
                            spt[2] = device_refCam.extrinsic[2 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3 + 2] * depth;
                            spt[3] = 1.0;
                            TCam projected[2];
                            device_cameras[v].projectPoint(spt, projected);
                            projected[0] /= downsample;
                            projected[1] /= downsample;
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
                        count += 3;
                    }
                }

                //if the overlap pixels of two patches are too few, skip this frame
                if(count < patchSize * 3 / 2)
                    continue;

                mean1 /= count;
                mean2 /= count;

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
                    var1 = sqrt(var1 / (count - 1));
                    var2 = sqrt(var2 / (count - 1));
                    TOut ncc = 0;
                    for (int i = 0; i < patchSize; ++i) {
                        if (newPatch[3 * i] >= 0 && refPatch[3 * i] >= 0) {
                            ncc += (refPatch[3 * i] - mean1) * (newPatch[3 * i] - mean2) +
                                   (refPatch[3 * i + 1] - mean1) * (newPatch[3 * i + 1] - mean2) +
                                   (refPatch[3 * i + 2] - mean1) * (newPatch[3 * i + 2] - mean2);
                        }
                    }
                    nccArray[v] = ncc / (var1 * var2 * (count - 1));
                }
            }

            int validCount = 0;
            for(int i=0; i<N; ++i){
                if(nccArray[i] >= 0)
                    nccValid[validCount++] = nccArray[i];
            }

            //median of truncate NCC
            //truncate value
            const TOut thetancc = 0.3;
            //if not visible in over 50% frames, assign large penalty
            if(validCount < 2){
                output[outputOffset] = 1;
            }else{
                insert_sort<TOut>(nccValid, validCount);
                int kth = validCount / 2;
                TOut res = 0;
                for(int i=kth; i < validCount; ++i){
                    if(nccValid[i] < thetancc)
                        res += thetancc;
                    else
                        res += nccValid[i];
                }
                output[outputOffset] = 1.0 - res / (TOut)kth;
            }
        }
    }

    template<typename TCam, typename TOut>
    TOut stereoMatchingKernelDebug(const unsigned char* images, const unsigned char* refImage, const int width, const int height, const int N,
                                   const CudaCamera<TCam>* cameras, const CudaCamera<TCam>* refCam,
                                   const TCam* rays, const TCam min_disp, const TCam max_disp, const TCam downsample, const int resolution, const int R,
                                   const int x, const int y, const int d){
        TOut refPatch[MAXPATCHSIZE * 3];
        const int patchSize = (2*R+1) * (2*R+1);
        printf("Debuging on (%d,%d,%d), R=%d\n", x, y, (int)d, R);
        //the first thread in the block create reference patch
        int ind = 0;
        for (int dx = -1 * R; dx <= R; dx++) {
            for (int dy = -1 * R; dy <= R; dy++) {
                int curx = x + dx, cury = y + dy;
                printf("dx:%d, dy:%d, curx:%d, cury:%d\n", dx, dy, curx, cury);
                if (curx >= 0 && curx < width && cury >= 0 && cury < height) {
                    refPatch[3 * ind] = (TOut)refImage[3 * (cury * width + curx)];
                    refPatch[3 * ind + 1] = (TOut)refImage[3 * (cury * width + curx) + 1];
                    refPatch[3 * ind + 2] = (TOut)refImage[3 * (cury * width + curx) + 2];
                }else{
                    refPatch[3 * ind] = -1;
                    refPatch[3 * ind + 1] = -1;
                    refPatch[3 * ind + 2] = -1;
                }
                ind++;
            }
        }

        //allocate memory
        TOut nccArray[MAXFRAME] = {-1};
        TOut nccValid[MAXFRAME] = {};

        TOut newPatch[MAXPATCHSIZE * 3];


        //the number of threads in each block can be fewer than resolution
        //position inside output array
        for(auto i=0; i<MAXFRAME; ++i)
            nccArray[i]  = -1;

        TCam depth = 1.0/(min_disp + d * (max_disp - min_disp) / (TCam) resolution);

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
                        //project points. Be careful with the downsample factor
                        TCam spt[4];
                        spt[0] = refCam->extrinsic[0 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3] * depth;
                        spt[1] = refCam->extrinsic[1 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3 + 1] * depth;
                        spt[2] = refCam->extrinsic[2 + CudaCamera<TCam>::POSITION] + rays[(cury * width + curx) * 3 + 2] * depth;
                        spt[3] = 1.0;
                        TCam projected[2];
                        cameras[v].projectPoint(spt, projected);
                        projected[0] /= downsample;
                        projected[1] /= downsample;
                        if(v == 0 && dx == 0 && dy == 0){
                            printf("Camera pos:(%.3f,%.3f,%.3f)\n", refCam->extrinsic[0], refCam->extrinsic[1], refCam->extrinsic[2]);
                            printf("Camera ori:(%.3f,%.3f,%.3f)\n", refCam->extrinsic[3], refCam->extrinsic[4], refCam->extrinsic[5]);
                            printf("ray:(%.3f,%.3f,%.3f)\n", rays[(cury * width + curx) * 3], rays[(cury * width + curx) * 3 + 1], rays[(cury * width + curx) * 3 + 2]);
                            printf("3d point: (%.3f,%.3f,%.3ff)\nprojected point: (%.3f,%.3f)\n", spt[0], spt[1], spt[2], projected[0], projected[1]);
                        }
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
                    count += 3;
                }
            }

            //if the overlap pixels of two patches are too few, skip this frame
            if(count < patchSize * 3 / 2)
                continue;

            mean1 /= count;
            mean2 /= count;

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
                var1 = sqrt(var1 / (count - 1));
                var2 = sqrt(var2 / (count - 1));
                TOut ncc = 0;
                for (int i = 0; i < patchSize; ++i) {
                    if (newPatch[3 * i] >= 0 && refPatch[3 * i] >= 0) {
                        ncc += (refPatch[3 * i] - mean1) * (newPatch[3 * i] - mean2) +
                               (refPatch[3 * i + 1] - mean1) * (newPatch[3 * i + 1] - mean2) +
                               (refPatch[3 * i + 2] - mean1) * (newPatch[3 * i + 2] - mean2);
                    }
                }
                nccArray[v] = ncc / (var1 * var2 * (count - 1));
            }
        }

        int validCount = 0;
        for(int i=0; i<N; ++i){
            if(nccArray[i] >= 0)
                nccValid[validCount++] = nccArray[i];
        }

        //median of truncate NCC
        //truncate value
        const TOut thetancc = 0.3;
        //if not visible in over 50% frames, assign large penalty
        if(validCount < 2){
            return (TOut)1.0;
        }else{
            insert_sort<TOut>(nccValid, validCount);
            int kth = validCount / 2;
            TOut res = 0;
            for(int i=kth; i<validCount; ++i){
                if(nccValid[i] < thetancc)
                    res += thetancc;
                else
                    res += nccValid[i];
            }
            return 1.0 - res / (TOut)kth;
        }
    }
}//namespace CudaVision
#endif //DYNAMICSTEREO_CUSTEREOMATCHING_H
