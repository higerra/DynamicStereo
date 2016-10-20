//
// Created by yanhang on 10/5/16.
//

#ifndef DYNAMICSTEREO_TYPES_H
#define DYNAMICSTEREO_TYPES_H

namespace dynamic_stereo{
    namespace video_segment{
        enum PixelFeature{
            PIXEL_VALUE,
            PIXEL_HISTOGRAM,
            BRIEF
        };

        enum TemporalFeature{
            TRANSITION_PATTERN,
            HISTOGRAM,
            COMBINED
        };

    }//namespace video_segmentation
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TYPES_H
