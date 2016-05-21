//
// Created by yanhang on 5/21/16.
//

#ifndef DYNAMICSTEREO_DESCRIPTOR_H
#define DYNAMICSTEREO_DESCRIPTOR_H

#include <vector>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

namespace dynamic_stereo {

    namespace Feature {
        enum FeatureType {
            RGB_CAT
        };

        class FeatureConstructor {
        public:
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const = 0;
            void normalizel2(std::vector<float> &array) const;

            inline int getDim() const{return dim;}
        protected:
            int dim;
        };

        class RGBCat : public FeatureConstructor {
        public:
            RGBCat(const int kBin_ = 8, const float min_diff_ = 10) : kBin(kBin_), kBinIntensity(kBin/2), min_diff(min_diff_), cut_thres(0.1){
                CHECK_GT(kBin, 0);
                CHECK_GT(kBinIntensity, 0);
                binUnit = 512 / (float) kBin;
                binUnitIntensity = 256 / (float) kBinIntensity;
                dim = kBin * 3 + kBinIntensity;
            }

            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const;
        private:
            const int kBin;
            const int kBinIntensity;
            float binUnit;
            float binUnitIntensity;
            const float min_diff;
            const float cut_thres;
        };

    }//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DESCRIPTOR_H
