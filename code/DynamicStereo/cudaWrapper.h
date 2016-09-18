//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUDAWRAPPER_H
#define DYNAMICSTEREO_CUDAWRAPPER_H
#include <vector>
#include <glog/logging.h>

namespace dynamic_stereo {

	using TCam = double;
	using TOut = float;

	bool checkDevice();

	void callStereoMatching(const std::vector<unsigned char>& images, const std::vector<unsigned char>& refImage,
	                        const int width, const int height, const int N,
                            const TCam min_disp, const TCam max_disp, const TCam downsample,
	                        const std::vector<TCam>& intrinsics, const std::vector<TCam>& extrinsics,
                            const std::vector<TCam>& refInt, const std::vector<TCam>& refExt, const std::vector<TCam>& spts,
	                        const int resolution, const int R,
	                        std::vector<TOut>& result);
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_CUDAWRAPPER_H
