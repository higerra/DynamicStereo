//
// Created by yanhang on 7/29/16.
//

#ifndef DYNAMICSTEREO_PIXEL_FEATURE_H
#define DYNAMICSTEREO_PIXEL_FEATURE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <memory>
#include "distance_metric.h"

namespace cv{
    namespace xfeatures2d {
        class BriefDescriptorExtractor;
    }
}

namespace dynamic_stereo {
    namespace video_segment {
        using VideoMat = std::vector<cv::Mat>;

        enum PixelFeature{
            PIXEL,
            BRIEF
        };

        enum TemporalFeature{
            TRANSITION_PATTERN,
            TRANSITION_COUNTING
        };

        class FeatureBase {
        public:
            inline const DistanceMetricBase *getDefaultComparator() const {
                return comparator.get();
            }

            inline void setComparator(std::shared_ptr<DistanceMetricBase> new_comparator) {
                CHECK(new_comparator.get());
                comparator.reset();
                comparator = new_comparator;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray &output) const = 0;

            inline int getDim() const { return dim; }

            virtual void printFeature(const cv::InputArray input) const {
                printf("Not implemented yet\n");
            }

        protected:
            std::shared_ptr<DistanceMetricBase> comparator;
            int dim;
        };

        /////////////////////////////////////////////////////////////
        //pixel level feature
        class PixelFeatureExtractorBase : public FeatureBase {
        public:
            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const {}
        };

        class PixelValue : public PixelFeatureExtractorBase {
        public:
            PixelValue() {
                comparator.reset(new DistanceL2());
                FeatureBase::dim = 3;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;
        };

        class BRIEFWrapper : public PixelFeatureExtractorBase {
        public:
            BRIEFWrapper();

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;

        private:
            cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> cvBrief;
        };

        ////////////////////////////////////////////////////////////
        //temporal feature
        class TemporalFeatureExtractorBase : public FeatureBase {
        public:
            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const {}

            //some pixel feature algorithms achieve significant speed up when compute at image level, the below routine
            //use precomputed pixel features as input.
            //pixelFeatures: precomputed pixel features. Feature Mats are flattened. Each frame is a (w*h) by k Mat
            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const = 0;
        };

        class TransitionFeature : public TemporalFeatureExtractorBase {
        public:
            TransitionFeature(const PixelFeatureExtractorBase *pf_, const int s1_, const int s2_, const float theta_) :
                    pixel_feature(pf_), pixel_distance(CHECK_NOTNULL(pf_->getDefaultComparator())), s1(s1_), s2(s2_),
                    t(theta_) {
                CHECK_GT(stride1(), 0);
                CHECK(pixel_distance);
            }

            inline int stride1() const { return s1; }

            inline int stride2() const { return s2; }

            inline float theta() const { return t; }

            inline const PixelFeatureExtractorBase *getPixelFeatureExtractor() const {
                return pixel_feature;
            }

            inline const DistanceMetricBase *getPixelFeatureComparator() const {
                return pixel_distance;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const = 0;

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const = 0;

        protected:
            const PixelFeatureExtractorBase *pixel_feature;
            const DistanceMetricBase *pixel_distance;
            const int s1;
            const int s2;
            const float t;
        };

        class TransitionPattern : public TransitionFeature {
        public:
            TransitionPattern(const PixelFeatureExtractorBase *pf_,
                              const int s1_, const int s2_, const float theta_) :
                    TransitionFeature(pf_, s1_, s2_, theta_), binPerCell(8) {
                comparator.reset(new DistanceHammingAverage());
                or_table.resize(4);
                or_table[0] = 0;
                or_table[1] = 2;
                or_table[2] = 4;
                or_table[3] = 8;
            }

            virtual void extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const;

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input);

        private:
            int getKChannel(const int kFrames) const;

            std::vector<uchar> or_table;
            const int binPerCell;
        };

        class TransitionCounting : public TransitionFeature {
        public:
            TransitionCounting(const PixelFeatureExtractorBase *pf_,
                               const int s1_, const int s2_, const float theta_)
                    : TransitionFeature(pf_, s1_, s2_, theta_) {
                comparator.reset(new DistanceL2());
            }

            virtual void extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const;

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                 cv::OutputArray feats) const;
        };


    }//namespace video_segment
}
#endif //DYNAMICSTEREO_PIXEL_FEATURE_H
