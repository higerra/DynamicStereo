//
// Created by yanhang on 9/16/16.
//

#include "cudaWrapper.h"
#include <cuda_runtime.h>
#include "../CudaVision/cuStereoMatching.h"
#include <vector>

namespace dynamic_stereo{

	bool checkDevice(){
        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        LOG(INFO) << nDevices << " Cuda compatiable GPUs found";

        bool device_found = false;
        for(int i=0; i<nDevices; ++i){

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("Device %d, major version: %d, totalGlobalMem: %zu", i, prop.major, prop.totalGlobalMem);
            if(prop.major >= 5){
                LOG(INFO) << "Device " << prop.name << " meets requiredment.";
                cudaSetDevice(i);
                device_found = true;
                break;
            }
        }
		return device_found;
	}

	void callStereoMatching(const std::vector<unsigned char>& images, const std::vector<unsigned char>& refImage,
	                        const int width, const int height, const int N,
                            const TCam min_disp, const TCam max_disp, const TCam downsample,
	                        const std::vector<TCam>& intrinsics, const std::vector<TCam>& extrinsics,
                            const std::vector<TCam>& refInt, const std::vector<TCam>& refExt,
                            const std::vector<TCam>& rays, const int resolution, const int R,
	                        std::vector<TOut>& result) {
        CHECK_LE((2*R+1) * (2*R+1), CudaVision::MAXPATCHSIZE) << "The patch size can not be greater than 3.";
        CHECK_LE(N, CudaVision::MAXFRAME);
        CHECK_EQ(result.size(), width * height * resolution);

        LOG(INFO) << "Constructing CudaCamera";
        std::vector<CudaVision::CudaCamera<TCam> > cuCameras(N);
        CudaVision::CudaCamera<TCam> refCam;

        const int &kIntrinsic = CudaVision::CudaCamera<TCam>::kIntrinsicSize;
        const int &kExtrinsic = CudaVision::CudaCamera<TCam>::kExtrinsicSize;
        CHECK_EQ(intrinsics.size(), N * kIntrinsic);
        CHECK_EQ(extrinsics.size(), N * kExtrinsic);
        for (auto i = 0; i < kIntrinsic; ++i) {
            for (auto v = 0; v < N; ++v)
                cuCameras[v].intrinsic[i] = intrinsics[kIntrinsic * v + i];
            refCam.intrinsic[i] = refInt[i];
        }
        for (auto i = 0; i < kExtrinsic; ++i) {
            for (auto v = 0; v < N; ++v)
                cuCameras[v].extrinsic[i] = extrinsics[kExtrinsic * v + i];
            refCam.extrinsic[i] = refExt[i];
        }

        LOG(INFO) << "Calling kernel";
        //calling kernel
        if (width == 960 && height == 540) {
            CudaVision::CudaStereoMatching<TCam, TOut, 960, 540> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else if (width == 540 && height == 960) {
            CudaVision::CudaStereoMatching<TCam, TOut, 540, 960> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else if (width == 640 && height == 360) {
            CudaVision::CudaStereoMatching<TCam, TOut, 640, 360> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else if (width == 360 && height == 640) {
            CudaVision::CudaStereoMatching<TCam, TOut, 360, 640> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else if (width == 480 && height == 270) {
            CudaVision::CudaStereoMatching<TCam, TOut, 480, 270> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else if (width == 320 && height == 180) {
            CudaVision::CudaStereoMatching<TCam, TOut, 320, 180> matching(N, resolution, R, min_disp, max_disp, downsample);
            matching.run(images.data(), refImage.data(), cuCameras.data(), &refCam, rays.data(), result);
        } else {
            printf("Unsupprted size\n");
            CHECK(true) << "Unsupported image size: " << width << ' ' << height;
        }

        LOG(INFO) << "Done";
    }
}//namespace dynamic_stereo

