//
// Created by yanhang on 5/15/16.
//

#ifndef DYNAMICSTEREO_TEMPMEANSHIFT_H
#define DYNAMICSTEREO_TEMPMEANSHIFT_H
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>

namespace dynamic_stereo{

	class DataSet {
	public:
		using FeatureSet = std::vector<std::vector<std::vector<float> > >;
		inline const FeatureSet& getFeatures() const{
			return features;
		}
		inline FeatureSet& getFeatures(){
			return features;
		}
		void appendDataSet(const DataSet& newData);

		void dumpData_libsvm(const std::string& path) const;
		void dumpData_table(const std::string& path) const{}

		void printStat() const{
			if(features.size() == 1){
				printf("Number of samples: %d\n", (int)features[0].size());
			}else if(features.size() == 2) {
				float kPos = (float) features[1].size();
				float kNeg = (float) features[0].size();
				CHECK_GT(kPos+kNeg, 1);
				printf("Positive: %d(%.2f), negative: %d(%.2f)\n", (int)kPos, kPos/(kPos+kNeg), (int)kNeg, kNeg/(kPos+kNeg));
			}
		}
	private:
		FeatureSet features;
	};

	class FeatureConstructor{
	public:
		FeatureConstructor(const int kBin_ = 8, const float min_diff_ = 10): kBin(kBin_), kBinIntensity(kBin_/2), min_diff(min_diff_), cut_thres(0.25){
			CHECK_GT(kBin, 0);
			CHECK_GT(kBinIntensity, 0);
			binUnit = 512 / (float)kBin;
			binUnitIntensity = 256 / (float)kBinIntensity;
		}
		virtual void constructFeature(const std::vector<float>& array, std::vector<float>& feat) const = 0;

		void normalizel2(std::vector<float>& array) const;
	protected:
		const int kBin;
		const int kBinIntensity;
		float binUnit;
		float binUnitIntensity;
		const float min_diff;
		const float cut_thres;
	};

	class RGBCat: public FeatureConstructor{
	public:
		RGBCat(const int kBin_ = 8, const float min_diff_ = 10): FeatureConstructor(kBin_, min_diff_){}
		virtual void constructFeature(const std::vector<float>& array, std::vector<float>& feat) const;
	};

	namespace Feature {
		enum FeatureType {RGB_CAT};
		cv::Size importData(const std::string &path, std::vector<std::vector<float> > &array, const int downsample,
							const int tWindow);

		//samples: when extracting training samples (gt is not empty), samples have the size 2;
		// when extracing testing samples (gt is empty), samples have the size 1
		void extractFeature(const std::vector<std::vector<float> > &array, const cv::Size &dims, const cv::Mat &gt,
							DataSet& samples, const int kBin, const float min_diff, const FeatureType method);
	}

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TEMPMEANSHIFT_H
