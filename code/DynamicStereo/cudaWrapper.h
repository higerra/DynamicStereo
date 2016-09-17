//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUDAWRAPPER_H
#define DYNAMICSTEREO_CUDAWRAPPER_H
#include <vector>

namespace dynamic_stereo {

	using TCam = double;
	using TOut = float;

	bool checkDevice();

	void callStereoMatching(const std::vector<unsigned char>& images, const std::vector<unsigned char>& refImage,
	                        const int width, const int height, const int N,
	                        const std::vector<TCam>& intrinsics, const std::vector<TCam>& extrinsics,
	                        const int resolution, const int R,
	                        std::vector<TOut>& result);
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_CUDAWRAPPER_H
