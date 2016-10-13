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

            inline std::shared_ptr<DistanceMetricBase> getDefaultComparatorSmartPointer() {
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

            /// Get the dimension of the descriptor
            /// \return the dimension of the descriptor
            inline int getDim() const { return dim_; }

            /// Print the content of descriptor for debugging purpose. The default implementation does noting
            /// \param input
            virtual void printFeature(const cv::InputArray input) const = 0;

        protected:
            std::shared_ptr<DistanceMetricBase> comparator_;
            int dim_;
        };

        class PixelFeatureExtractorBase: public FeatureBase{
        public:
            /*!
             * Extract descriptors for all pixels of a image.
             * Some algorithm will achieve great speed up when computing descriptor for all pixels (like BRIEF)
             * @param input Input image
             * @param output Output feature descriptor
             */
            virtual void extractAll(const cv::InputArray input, cv::OutputArray &output) const = 0;
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

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;

            virtual void printFeature(const cv::InputArray input) const{
                std::cout << input.getMat() << std::endl;
            }
        };

        ///
        ///A wrapper class for OpenCV's BRIEF feature descriptor
        ///
        class BRIEFWrapper : public PixelFeatureExtractorBase {
        public:
            BRIEFWrapper();

            virtual void extractAll(const cv::InputArray input, cv::OutputArray output) const;

            virtual void printFeature(const cv::InputArray input) const{
                std::cerr << "Not implemented" << std::endl;
            }
        private:
            cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> cvBrief;
        };


        //////////////////////////////////////////////////////////////////////////
        /*!
         * Descriptor for temporal features.
         * Temporla features are extracted from a set of pixels (x,y,0),(x,y,1)...(x,y,N), where N is the number of frames
         */
        class TemporalFeatureExtractorBase : public FeatureBase {
        public:
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
         * Simple average temporal descriptor
         */
        class TemporalAverage: public TemporalFeatureExtractorBase{
        public:
            TemporalAverage() {
                FeatureBase::comparator_.reset(new DistanceL2());
                FeatureBase::dim_ = 3;
            }

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input) const{
                std::cout << input.getMat() << std::endl;
            }
        };


        ///
        ///Build a HSV histogram from local color
        ///
        class ColorHistogram: public TemporalFeatureExtractorBase{
        public:
            enum ColorSpace{
                BGR,
                HSV,
                LAB
            };
            ColorHistogram(const ColorSpace cspace, const std::vector<int>& kBin,
                           const int width, const int height, const int R);

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input) const{
                std::cout << input.getMat() << std::endl;
            }
        private:
            std::vector<int> kBin_;
            std::vector<float> bin_unit;
            std::vector<float> chn_offset;

            const int width_;
            const int height_;
            const int R_;
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
            TransitionFeature(const int kFrames, const int s1, const int s2, const float theta,
                              const DistanceMetricBase* pixel_distance) :
                    pixel_distance_(CHECK_NOTNULL(pixel_distance)), kFrames_(kFrames), s1_(s1), s2_(s2), t_(theta) {}
            inline int stride1() const { return s1_; }

            inline int stride2() const { return s2_; }

            inline float theta() const { return t_; }

            inline const DistanceMetricBase *getPixelFeatureComparator() const {
                return pixel_distance_;
            }

            inline int GetKFrames() const{return kFrames_;}
            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const = 0;

        protected:
            const DistanceMetricBase *pixel_distance_;
            const int kFrames_;
            const int s1_;
            const int s2_;
            const float t_;
        };

        /*!
         * Temporal transition pattern consists of two parts: firstly we compare pixel (x,y,i) with (x,y,i+stride1) and get
         * a binary bin based on whether the pixel features changes a lot; secondly we compare (x,y,i) with (x,y,N/2+i) and
         * get the same binary bin. The binary descriptor is compressed into uchar using bit operation
         */
        class TransitionPattern : public TransitionFeature {
        public:
            /*!
             * Constructor
             * \sa TransitionFeature
             */
            TransitionPattern(const int kFrames, const int s1, const int s2, const float theta,
                              const DistanceMetricBase* pixel_distance) :
                    TransitionFeature(kFrames, s1, s2, theta, pixel_distance), binPerBlock_(8) {
                comparator_.reset(new DistanceHammingAverage());
                or_table_.resize((size_t)binPerBlock_, (uchar)0);
                or_table_[0] = (uchar)1;
                for(auto i=1; i<binPerBlock_; ++i){
                    or_table_[i] = or_table_[i-1] << 1;
                }
                FeatureBase::dim_ = getKChannel(TransitionFeature::kFrames_);
            }

            virtual void computeFromPixelFeature(const cv::InputArray pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input) const;

        private:
            int getKChannel(const int kFrames) const;
            std::vector<uchar> or_table_;
            const int binPerBlock_;
        };

        /*!
         * Combined descriptor for both temporal transition and appearance
         * Use DistanceCombinedWeighting for distance metric
         */
        class CombinedTemporalFeature: public TemporalFeatureExtractorBase{
        public:
            /*!
             * Constructor
             * @return
             */
            CombinedTemporalFeature(const std::vector<std::shared_ptr<TemporalFeatureExtractorBase> > extractors,
                                    const std::vector<double>& weights,
                                    const std::vector<std::shared_ptr<DistanceMetricBase> >* sub_comparators = nullptr);


            inline const std::vector<double> & GetSubWeights() const{
                return weights_;
            }

            virtual void
            extractPixel(const cv::InputArray input, const int x, const int y, cv::OutputArray output) const {
                CHECK(true) << "Not implemented yet";
            }

            virtual void computeFromPixelFeature(const cv::InputArray pixel_features, cv::OutputArray feats) const{
                CHECK(true) << "This method shouldn't be called";
            }

            void computeFromPixelFeatures(const std::vector<std::vector<cv::Mat> >& pixelFeatures, cv::OutputArray feats) const;

            virtual void printFeature(const cv::InputArray input) const;
        private:
            std::vector<std::shared_ptr<TemporalFeatureExtractorBase> > temporal_extractors_;
            std::vector<std::shared_ptr<DistanceMetricBase> > sub_comparators_;
            std::vector<int> offset_;
            std::vector<double> weights_;
        };

    }//namespace video_segment
}
#endif //DYNAMICSTEREO_PIXEL_FEATURE_H
