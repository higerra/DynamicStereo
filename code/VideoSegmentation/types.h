//
// Created by yanhang on 10/5/16.
//

#ifndef DYNAMICSTEREO_TYPES_H
#define DYNAMICSTEREO_TYPES_H

namespace dynamic_stereo{
    namespace video_segment{
        enum PixelFeature{
            PIXEL,
            BRIEF
        };

        enum TemporalFeature{
            TRANSITION_PATTERN,
            TRANSITION_COUNTING,
            TRANSITION_AND_APPEARANCE
        };
    }//namespace video_segmentation
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TYPES_H