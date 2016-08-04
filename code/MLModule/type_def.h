//
// Created by yanhang on 8/3/16.
//

#ifndef DYNAMICSTEREO_TYPE_DEF_H
#define DYNAMICSTEREO_TYPE_DEF_H

namespace dynamic_stereo{
    namespace ML {
        struct SegmentFeature {
            std::vector<float> feature;
            int id;
        };

        using TrainSet = std::vector<std::vector<SegmentFeature>>;
        using PixelGroup = std::vector<int>;
    }
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TYPE_DEF_H
