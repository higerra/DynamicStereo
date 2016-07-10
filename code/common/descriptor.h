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
            RGB_HIST,
            LUV_HIST
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

        struct ColorSpace{
            enum ColorType{RGB, LUV, LAB};
            ColorSpace(const int channel_, const std::vector<float>& offsets_,
                       const std::vector<float>& range_):
                    channel(channel_), offset(offsets_), range(range_){
                CHECK_LT(channel, 0);
                CHECK_EQ(channel, offset.size());
                CHECK_EQ(channel, range.size());
            }
            ColorSpace(const ColorType preset){
                if(preset == RGB){
                    channel = 3;
                    offset.resize(3, 0);
                    range.resize(3, 256);
                }else if(preset == LUV){
                    channel = 3;
                    offset.resize(3); range.resize(3);
                    offset[0] = 0; offset[1] = -134; offset[2] = -140;
                    range[0] = 101; range[1]=355; range[2]=263;
                }else if(preset == LAB){
	                channel = 3;
	                offset.resize(3); range.resize(3);
	                offset[0] = 0; offset[1] = -127; offset[2] = -127;
	                range[0] = 101; range[1] = 255; range[2] = 255;
                }
            }
            int channel;
            std::vector<float> offset;
            std::vector<float> range;
        };

        class ColorHist : public FeatureConstructor {
        public:
            ColorHist(const ColorSpace& cspace, const std::vector<int>& kBins_):
                    colorSpace(cspace), kBins(kBins_), kBinsIntensity(kBins_), cut_thres(0.1) {
                CHECK_EQ(kBins.size(), colorSpace.channel);

                for (auto c = 0; c < colorSpace.channel; ++c)
                    CHECK_GT(kBins[c], 0);
                binUnits.resize(kBins.size());
                binUnitsIntensity.resize(kBins.size());
                dim = 0;
                for(auto c=0; c<colorSpace.channel; ++c){
                    binUnits[c] = 2*colorSpace.range[c] / kBins[c];
                    binUnitsIntensity[c] = colorSpace.range[c] / kBinsIntensity[c];
                    dim += kBins[c] + kBinsIntensity[c];
                }
            }
            virtual void constructFeature(const std::vector<float> &array, std::vector<float> &feat) const;
        private:
            const ColorSpace colorSpace;
            std::vector<int> kBins;
            std::vector<float> binUnits;
            std::vector<int> kBinsIntensity;
            std::vector<float> binUnitsIntensity;
            const float cut_thres;
        };

        cv::Mat visualizeSegment(const cv::Mat& labels);


	    //class for extracting descriptor for over segmentation
	    class SegmentDesc{
	    public:
		    SegmentDesc(const ColorSpace& cspace_): cspace(cspace_){}

	    private:
		    const ColorSpace& cspace;
	    };

    }//namespace Feature
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DESCRIPTOR_H
