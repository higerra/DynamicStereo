//
// Created by yanhang on 3/28/16.
//

#ifndef DYNAMICSTEREO_GRIDWARPPING_H
#define DYNAMICSTEREO_GRIDWARPPING_H

#include <opencv/opencv2.hpp>
#include <Eigen/Eigen>
#include <theia/theia.h>
#include "model.h"

namespace dynamic_stereo {
    class GridWarpping {
    public:
        typedef int EnergyType;
        GridWarpping(const StereoModel<EnergyType>& model_, const theia::Reconstruction& reconstruction):
                model(model_), reconstruction(reconstruction_){}

    private:

        const StereoModel<EnergyType>& model;
        const theia::Reconstruction& reconstruction;
    };
}

#endif //DYNAMICSTEREO_GRIDWARPPING_H
