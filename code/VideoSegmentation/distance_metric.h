//
// Created by yanhang on 7/29/16.
//

#ifndef DYNAMICSTEREO_DISTANCE_METRIC_H
#define DYNAMICSTEREO_DISTANCE_METRIC_H

#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <glog/logging.h>
#include <limits>
#include <string>
#include <bitset>
#include <opencv2/opencv.hpp>

namespace dynamic_stereo{

    class DistanceMetricBase{
    public:
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const = 0;
    };

    class DistanceL1: public DistanceMetricBase{
    public:
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const {
            return cv::norm(a1, a2, cv::NORM_L1);
        }
    };

    class DistanceL1Average: public DistanceMetricBase{
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const {
            return cv::norm(a1,a2,cv::NORM_L1) / (float)(a1.rows() * a1.cols());
        }
    };

    class DistanceL2: public DistanceMetricBase{
    public:
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const {
            return cv::norm(a1, a2, cv::NORM_L2);
        }
    };

    //to use efficient std::bitset, the length of the array must be known at compile time (in byte)
	class DistanceHamming: public DistanceMetricBase{
	public:
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const{
            return cv::norm(a1, a2, cv::NORM_HAMMING);
        }
	};

	class DistanceHammingAverage: public DistanceMetricBase{
	public:
        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const{
            const double len = (double)(a1.cols() * a1.rows() * 8);
            CHECK_GT(len, 0);
            double diff = cv::norm(a1, a2, cv::NORM_HAMMING);
            return diff / len;
        }
	};
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DISTANCE_METRIC_H