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

namespace dynamic_stereo{

    template<typename T, typename R = float>
    class DistanceMetricBase{
    public:
        virtual R evaluate(const std::vector<T>& a1, const std::vector<T>& a2) const = 0;
    };

    template<typename T, typename R = float>
    class DistanceL1: public DistanceMetricBase<T, R>{
    public:
        virtual R evaluate(const std::vector<T>& a1, const std::vector<T>& a2) const {
            CHECK_EQ(a1.size(), a2.size());
            CHECK(!a1.empty());
            std::vector <T> diff(a1.size());
            std::transform(a1.begin(), a1.end(), a2.begin(), diff.begin(),
                           [](const T &a, const T &b) { return std::abs(a - b); });
            return static_cast<R>(std::accumulate(diff.begin(), diff.end(), (T) 0));
        }
    };

    template<typename T, typename R = float>
    class DistanceL1Average: public DistanceMetricBase<T, R>{
        virtual R evaluate(const std::vector<T>& a1, const std::vector<T>& a2) const {
            DistanceL1<T, R> l1dis;
            R res = static_cast<R>(l1dis.evaluate(a1, a2) / (float)a1.size());
            return res;
        }
    };

    template<typename T, typename R = float>
    class DistanceL2: public DistanceMetricBase<T, R>{
    public:
        virtual R evaluate(const std::vector<T>& a1, const std::vector<T>& a2) const {
            CHECK_EQ(a1.size(), a2.size());
            CHECK(!a1.empty());
            std::vector <T> diff(a1.size());
            std::transform(a1.begin(), a1.end(), a2.begin(), diff.begin(),
                           [](const T &a, const T &b) { return (a - b) * (a - b); });
            return static_cast<R>(
                    std::sqrt(std::accumulate(diff.begin(), diff.end(), (T) 0) + std::numeric_limits<T>::epsilon()));
        }
    };

	class DistanceHamming: public DistanceMetricBase<bool, int>{
	public:
		virtual int evaluate(const std::vector<bool>& a1, const std::vector<bool>& a2) const {
			CHECK_EQ(a1.size(), a2.size());
			int res = 0;
			for(auto i=0; i<a1.size(); ++i){
				if(a1[i] xor a2[i])
					res++;
			}
			return res;
		}
	};

	class DistanceHammingAverage: public DistanceMetricBase<bool, float>{
	public:
		virtual float evaluate(const std::vector<bool>& a1, const std::vector<bool>& a2) const{
			CHECK(!a1.empty());
			DistanceHamming hamming;
			int dis = hamming.evaluate(a1, a2);
			return (float)dis / (float)a1.size();
		}
	};

}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DISTANCE_METRIC_H