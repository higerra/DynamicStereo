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

        class ColorHist: public FeatureConstructor{
        public:
            ColorHist(const int kBin_ = 8, const float min_diff_ = -1) : kBin(kBin_), min_diff(min_diff_), cut_thres(0.1){
                CHECK_GT(kBin, 0);
            }
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const = 0;
        protected:
            const int kBin;
            float binUnit;
            const float min_diff;
            const float cut_thres;
        };

        class RGBHist : public ColorHist {
        public:
            RGBHist(const int kBin_ = 8, const float min_diff_ = -1) : ColorHist(kBin_, min_diff_), kBinIntensity(kBin_){
                CHECK_GT(kBinIntensity, 0);
                binUnit = 512 / (float) kBin;
                binUnitIntensity = 256 / (float) kBinIntensity;
                dim = (kBin + kBinIntensity) * 3;
            }
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const;
        private:
            const int kBinIntensity;
            float binUnitIntensity;
        };

        cv::Mat visualizeSegment(const cv::Mat& labels);

        void meanshiftCluster(const cv::Mat& input, cv::Mat& output, const int hs, const float hr, const int min_a);

        void clusterRGBHist(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster, const int kBin = 8);

        void clusterRGBStat(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster);

    }//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DESCRIPTOR_H
