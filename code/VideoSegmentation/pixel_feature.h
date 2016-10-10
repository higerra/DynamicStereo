//
// Created by yanhang on 7/29/16.
//

#ifndef DYNAMICSTEREO_PIXEL_FEATURE_H
#define DYNAMICSTEREO_PIXEL_FEATURE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>
#include <memory>

#include "distance_metric.h"
#include "types.h"

namespace dynamic_stereo {
    namespace video_segment {
        using VideoMat = std::vector<cv::Mat>;

        ///
        ///Base class for all feature descriptors
        ///
        class FeatureBase {
        public:
            inline const DistanceMetricBase *getDefaultComparator() const {
                return comparator_.get();
            }

            inline std::shared_ptr<DistanceMetricBase> getDefaultComparatorPointer() {
                return comparator_;
            }
            /*!
             * Set the comparator for descriptors
             * @param new_comparator a shared pointer to new comparator
             */
            inline void setComparator(std::shared_ptr<DistanceMetricBase> new_comparator) {
                CHECK(new_comparator.get());
                comparator_.reset();
                comparator_ = new_comparator;
            }

            /*!
             * Extract descriptor for a pixel
             * @param input Input image
             * @param x x coordinate
             * @param y y corrdinate
             * @param output Output Mat for descriptor
             */
            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            /*!
             * Extract descriptors for all pixels of a image.
             * Some algorithm will achieve great speed up when computing descriptor for all pixels (like BRIEF)
             * @param input Input image
             * @param output Output feature descriptor
             */
            virtual void extractAll(const cv::InputArray input, cv::OutputArray &output) const = 0;

            /// Get the dimension of the descriptor
            /// \return the dimension of the descriptor
            inline int getDim() const { return dim_; }

            /// Print the content of descriptor for debugging purpose. The default implementation does noting
            /// \param input
            virtual void printFeature(const cv::InputArray input) const {
                printf("Not implemented yet\n");
            }

        protected:
            std::shared_ptr<DistanceMetricBase> comparator_;
            int dim_;
        };

        ///
        ///Base class for pixel features
        ///
        class PixelFeatureExtractorBase : public FeatureBase {
        public:
            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const {}
        };

        ///
        ///Use pixel value as feature
        ///
        class PixelValue : public PixelFeatureExtractorBase {
        public:
            PixelValue() {
                comparator_.reset(new DistanceL2());
                FeatureBase::dim_ = 3;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;
        };

        ///
        ///A wrapper class for OpenCV's BRIEF feature descriptor
        ///
        class BRIEFWrapper : public PixelFeatureExtractorBase {
        public:
            BRIEFWrapper();

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;

        private:
            cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> cvBrief;
        };

        class PixelHistogram: public PixelFeatureExtractorBase{
        public:
            PixelHistogram(const std::vector<int>& kBin){

            }
        private:
            std::vector<int> kBin_;
        };

        /*!
         * Descriptor for temporal features.
         * Temporla features are extracted from a set of pixels (x,y,0),(x,y,1)...(x,y,N), where N is the number of frames
         */
        class TemporalFeatureExtractorBase : public FeatureBase {
        public:
            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const = 0;

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const {}

            /*!
             * Some pixel feature algorithms achieve significant speed up when compute at image level, the below routine
             * use precomputed pixel features as input.
             * pixelFeatures: precomputed pixel features. Feature Mats are flattened. Each frame is a (w*h) by k Mat
             * @param pixelFeatures Precomputed pixel features
             * @param feats Output temporal descriptors
             */
            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const = 0;
        };


        /*!
         * Descriptor based on temporal transition. Pixel transitions are evaluate with two stride: (x,y,i) with (x,y,i+stride1)
         * and (x,y,i) and (x,y,N/2+i)
         */
        class TransitionFeature : public TemporalFeatureExtractorBase {
        public:
            /*!
             * The constructor
             * @param pf_ Pixel feature extractor
             * @param s1_ Stride 1
             * @param s2_ Stride 2
             * @param theta_ Threshold
             * @return
             */
            TransitionFeature(const PixelFeatureExtractorBase *pf_, const int s1_, const int s2_, const float theta_) :
                    pixel_feature(pf_), pixel_distance(CHECK_NOTNULL(pf_->getDefaultComparator())), s1(s1_), s2(s2_),
                    t(theta_) {
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

        /*!
         * Temporal transition pattern consists of two parts: firstly we compare pixel (x,y,i) with (x,y,i+stride1) and get
         * a binary bin based on whether the pixel features changes a lot; secondly we compare (x,y,i) with (x,y,N/2+i) and
         * get the same binary bin.
         */
        class TransitionPattern : public TransitionFeature {
        public:
            /*!
             * Constructor
             * \sa TransitionFeature
             */
            TransitionPattern(const PixelFeatureExtractorBase *pf_,
                              const int s1_, const int s2_, const float theta_) :
                    TransitionFeature(pf_, s1_, s2_, theta_), binPerCell_(8) {
                comparator_.reset(new DistanceHammingAverage());
                or_table_.resize(4);
                or_table_[0] = 0;
                or_table_[1] = 2;
                or_table_[2] = 4;
                or_table_[3] = 8;
            }

            virtual void extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const;

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input);

        private:
            int getKChannel(const int kFrames) const;

            std::vector<uchar> or_table_;
            const int binPerCell_;
        };

        class TransitionCounting : public TransitionFeature {
        public:
            TransitionCounting(const PixelFeatureExtractorBase *pf_,
                               const int s1_, const int s2_, const float theta_)
                    : TransitionFeature(pf_, s1_, s2_, theta_) {
                comparator_.reset(new DistanceL2());
            }

            virtual void extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray feat) const;

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures,
                                                 cv::OutputArray feats) const;
        };

        /*!
         * Combined descriptor for both temporal transition and appearance
         * Use DistanceCombinedWeighting for distance metric
         */
        class TransitionAndAppearance: public TemporalFeatureExtractorBase{
        public:
            /*!
             * Constructor
             * @param pf_transition_ pixel descriptor used for computing transition
             * @param pf_appearance_ pixel descriptor used for computing appearance
             * @param s1_
             * @param s2_
             * @param theta_
             * @param weight_transition The weight of transition difference
             * @param weight_appearance The weight of appearcne difference
             * @return
             */
            TransitionAndAppearance(const PixelFeatureExtractorBase* pf_transition_,
                                    const PixelFeatureExtractorBase* pf_appearance_,
                                    const int s1_, const int s2_, const float theta_,
                                    const double weight_transition, const double weight_appearance);
            inline const std::vector<std::shared_ptr<DistanceMetricBase> >& GetSubComparators() const{
                return sub_comparators_;
            }
            inline const TransitionFeature* GetTransitionFeatureExtractor() const{
                return transition_feature_extractor_.get();
            }
            inline const std::vector<double> & GetSubWeights() const{
                return sub_weights_;
            }
            const size_t GetkBinAppearance() const{
                return kBinAppearance_;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const {
                CHECK(true) << "Not implemented yet";
            }

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const {}
            /*!
             * @param pixel_features_for_transition Precomputed pixel features for transition
             * @param pixel_features_for_appearance Precomputed pixel features for appearance
             * @param feats Output descriptor
             */
            void computeFromPixelAndAppearanceFeature(const cv::InputArray pixel_features_for_transition,
                                                      const cv::InputArray pixel_features_for_appearance,
                                                      cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input);
        private:
            const size_t kBinAppearance_;
            std::shared_ptr<TransitionFeature> transition_feature_extractor_;
            std::vector<std::shared_ptr<DistanceMetricBase> > sub_comparators_;
            std::vector<double> sub_weights_;
        };

    }//namespace video_segment
}
#endif //DYNAMICSTEREO_PIXEL_FEATURE_H
