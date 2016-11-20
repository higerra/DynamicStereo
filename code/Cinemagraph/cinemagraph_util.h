//
// Created by yanhang on 11/19/16.
//

#ifndef DYNAMICSTEREO_CINEMAGRAPH_UTIL_H
#define DYNAMICSTEREO_CINEMAGRAPH_UTIL_H

#include "cinemagraph.h"
namespace dynamic_stereo{
    namespace Cinemagraph{

        void ApproximateQuad(const std::vector<Eigen::Vector2i>& locs, const int width, const int height,
                             std::vector<int>& output, const bool refine);

    } //namespace Cinemagraph
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_CINEMAGRAPH_UTIL_H
