//
// Created by Yan Hang on 4/21/16.
//

#ifndef DYNAMICSTEREO_STABILIZATION_H
#define DYNAMICSTEREO_STABILIZATION_H

#include "../base/depth.h"
#include "../base/file_io"
#include "model.h"

namespace dynamic_stereo {
	class Stabilization {
	public:
		Stabilization(const FileIO& file_io_, const int offset_, const int tWindow);
		void runStabilization() const;
	private:
		const FileIO& file_io;
		const int offset;
	};

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_STABILIZATION_H
