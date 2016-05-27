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

        void normalizel2(std::vector<float> &array);
        void normalizeSum(std::vector<float> &array);

        class FeatureConstructor {
        public:
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const = 0;

            inline int getDim() const{return dim;}
        protected:
            int dim;
        };

        class RGBHist : public FeatureConstructor {
        public:
            RGBHist(const int kBin_ = 10, const float min_diff_ = -1) : kBin(kBin_), min_diff(min_diff_),
                                                                        kBinIntensity(kBin_), cut_thres(0.1){
                CHECK_GT(kBinIntensity, 0);
                binUnit = 512 / (float) kBin;
                binUnitIntensity = 256 / (float) kBinIntensity;
                dim = (kBin + kBinIntensity) * 3;
            }
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const;
        private:
            const int kBin;
            float binUnit;
            const float min_diff;
            const float cut_thres;
            const int kBinIntensity;
            float binUnitIntensity;
        };

        class LUVHist : public FeatureConstructor {
        public:
            LUVHist(const std::vector<int>& kBins_) : kBins(kBins_), kBinsIntensity(kBins_), cut_thres(0.1) {
                CHECK_EQ(kBins.size(), 3);
                for (auto c = 0; c < 3; ++c)
                    CHECK_GT(kBins[c], 0);
                binUnits.resize(kBins.size());
                binUnitsIntensity.resize(kBins.size());
                binUnits[0] = 200 / kBins[0];
                binUnits[1] = (220 + 134) * 2 / kBins[1];
                binUnits[2] = (122 + 140) * 2 / kBins[2];
                binUnitsIntensity[0] = 100 / kBins[0];
                binUnitsIntensity[1] = (220 + 134) / kBins[1];
                binUnitsIntensity[2] = (122 + 140) / kBins[2];
                dim = kBins[0] + kBins[1] + kBins[2] + kBinsIntensity[0] + kBinsIntensity[1] + kBinsIntensity[2];
            }
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const;
        private:
            std::vector<int> kBins;
            std::vector<float> binUnits;
            std::vector<int> kBinsIntensity;
            std::vector<float> binUnitsIntensity;
            const float cut_thres;
        };

        cv::Mat visualizeSegment(const cv::Mat& labels);

        void meanshiftCluster(const cv::Mat& input, cv::Mat& output, const int hs, const float hr, const int min_a);

        void clusterRGBHist(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster, const int kBin = 8);

        void clusterRGBStat(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster);

    }//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DESCRIPTOR_H
