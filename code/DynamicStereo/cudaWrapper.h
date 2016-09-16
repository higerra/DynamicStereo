//
// Created by yanhang on 9/16/16.
//

#ifndef DYNAMICSTEREO_CUDAWRAPPER_H
#define DYNAMICSTEREO_CUDAWRAPPER_H

namespace dynamic_stereo{

    using TCam = double;
    using TOut = float;

    void cudaCaller(const unsigned char* images, const unsigned char* refImage,
                    const int width, const int height, const int N,
                    const TCam* position, const TCam* axis,
                    const TCam min_disp, const TCam max_disp, const int resolution,
                    TOut* result);
}
#endif //DYNAMICSTEREO_CUDAWRAPPER_H
