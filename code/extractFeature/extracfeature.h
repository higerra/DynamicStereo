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
#include "descriptor.h"

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
		void dumpData_csv(const std::string& path) const;

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

	namespace Feature {
		cv::Size importData(const std::string &path, std::vector<std::vector<float> > &array, const int downsample,
							const int tWindow);
		cv::Size importDataMat(const std::string& path, std::vector<cv::Mat>& output, const int downsample, const int tWindow);
		//samples: when extracting training samples (gt is not empty), samples have the size 2;
		// when extracing testing samples (gt is empty), samples have the size 1
		void extractFeature(const std::vector<std::vector<float> > &array, const cv::Size &dims, const cv::Mat &gt,
							DataSet& samples, const int kBin, const float min_diff, const FeatureType method);
	}

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_TEMPMEANSHIFT_H
