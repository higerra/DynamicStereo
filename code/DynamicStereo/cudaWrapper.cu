//
// Created by yanhang on 9/16/16.
//

#include "cudaWrapper.h"
#include <cuda_runtime.h>
#include "../CudaVision/cuStereoMatching.h"
#include <vector>

namespace dynamic_stereo{

	bool checkDevice(){
		return true;
	}

	void callStereoMatching(const std::vector<unsigned char>& images, const std::vector<unsigned char>& refImage,
	                        const int width, const int height, const int N,
	                        const std::vector<TCam>& intrinsics, const std::vector<TCam>& extrinsics,
	                        const int resolution, const int R,
	                        std::vector<TOut>& result){

	}
}//namespace dynamic_stereo

