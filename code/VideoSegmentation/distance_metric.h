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

    class DistanceCombinedWeighting: public DistanceMetricBase{
    public:
        DistanceCombinedWeighting(const std::vector<size_t> splits,
                                  const std::vector<double> weights,
                                  const std::vector<std::shared_ptr<DistanceMetricBase> > comparators)
                : weights_(weights), comparators_(comparators){
            CHECK_EQ(splits.size() + 1, comparators_.size());
            CHECK_EQ(splits.size() + 1, weights_.size());
            ranges_.resize(splits.size() + 1);
            int index = 0;
            CHECK_GT(splits.front(), 0);
            ranges_[index++] = std::make_pair(0, splits.front());
            for(auto i = 0; i<splits.size() - 1; ++i){
                CHECK_GT(splits[i+1], splits[i]);
                ranges_[index++] = std::make_pair(splits[i], splits[i+1]);
            }
            ranges_[index] = std::make_pair(splits.back(), -1);
        }

        virtual double evaluate(const cv::InputArray a1, const cv::InputArray a2) const{
            cv::Mat a1m = a1.getMat();
            cv::Mat a2m = a2.getMat();

            double distance = 0.0;

            CHECK(a1m.cols == 1 || a1m.rows == 1) << "Input must be vectors";
            CHECK_EQ(a1m.cols, a2m.cols);
            CHECK_EQ(a1m.rows, a2m.rows);
            ranges_.back().second = std::max(a1m.cols, a1m.rows);
            CHECK_GE(ranges_.back().second, ranges_.back().first);

            for(auto i=0; i<ranges_.size(); ++i){
                if(a1m.rows == 1) {
                    distance += weights_[i] * CHECK_NOTNULL(comparators_[i].get())->evaluate(
                                        a1m.colRange(ranges_[i].first, ranges_[i].second),
                                        a2m.colRange(ranges_[i].first, ranges_[i].second));
                }else{
                    distance += weights_[i] * CHECK_NOTNULL(comparators_[i].get())->evaluate(
                            a1m.rowRange(ranges_[i].first, ranges_[i].second),
                            a2m.rowRange(ranges_[i].first, ranges_[i].second));
                }
            }

            return distance;
        }

    private:
        //position of splits
        const std::vector<double> weights_;
        mutable std::vector<std::pair<int, int> > ranges_;
        const std::vector<std::shared_ptr<DistanceMetricBase> > comparators_;
    };
}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_DISTANCE_METRIC_H