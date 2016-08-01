//
// Created by yanhang on 7/29/16.
//

#ifndef DYNAMICSTEREO_PIXEL_FEATURE_H
#define DYNAMICSTEREO_PIXEL_FEATURE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "distance_metric.h"

namespace cv{
    namespace xfeatures2d {
        class BriefDescriptorExtractor;
    }
}

namespace dynamic_stereo{
    using VideoMat = std::vector<cv::Mat>;

    /////////////////////////////////////////////////////////////
    //pixel level feature
    template<typename T>
    class PixelFeatureExtractorBase{
    public:
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<T>& feat) const = 0;
        virtual void extractImage(const cv::Mat& input, std::vector<std::vector<T> >& feats) const{
            CHECK(input.data);
            const int width = input.cols;
            const int height = input.rows;
            feats.resize((size_t)(width * height));
            for(auto y=0; y<height; ++y){
                for(auto x=0; x<width; ++x){
                    extractPixel(input, x,y, feats[y*width+x]);
                }
            }
        }
        virtual void extractImage(const cv::Mat& input, cv::OutputArray& output) const;
    };

    class PixelValue: public PixelFeatureExtractorBase<float>{
    public:
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<float>& feat) const;
        virtual void extractImage(const cv::Mat& input, cv::OutputArray& output) const;
    };

    class BRIEFWrapper: public PixelFeatureExtractorBase<uchar>{
    public:
        BRIEFWrapper();
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<uchar>& feat) const;
        virtual void extractImage(const cv::Mat& input, cv::OutputArray& output) const;
    private:
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> cvBrief;
    };

    ////////////////////////////////////////////////////////////
    //temporal feature
    template<typename T>
    class TemporalFeatureExtractorBase{
    public:
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<T>& feat) const = 0;

        //some pixel feature algorithms achieve significant speed up when compute at image level, the below routine
        //use precomputed pixel features as input.
        //pixelFeatures: precomputed pixel features. Feature Mats are flattened. Each frame is a (w*h) by k Mat
        virtual void computeFromPixelFeature(const VideoMat& pixelFeatures,
                                             std::vector<std::vector<T> >& feats) const = 0;
    };

	template<typename T>
    class TransitionFeature: public TemporalFeatureExtractorBase<T>{
    public:
	    using PixelType = float;
	    using FeatureType = T;
        TransitionFeature(const PixelFeatureExtractorBase<float>* pf_, const DistanceMetricBase<PixelType>* pd_,
                          const int s1_, const int s2_, const float theta_):
                pixel_feature(pf_), pixel_distance(pd_), s1(s1_), s2(s2_), t(theta_){
            CHECK_GT(stride1(), 0);
            CHECK(pixel_feature);
            CHECK(pixel_distance);
        }
        inline int stride1() const {return s1;}
        inline int stride2() const {return s2;}
        inline float theta() const {return t;}

        inline const PixelFeatureExtractorBase<PixelType>* getPixelFeatureExtractor() const{
            return pixel_feature;
        }
        inline const DistanceMetricBase<FeatureType>* getPixelFeatureComparator() const{
            return pixel_distance;
        }

	    virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<FeatureType>& feat) const = 0;

        virtual void computeFromPixelFeature(const VideoMat& pixelFeatures,
                                             std::vector<std::vector<T> >& feats) const = 0;
    protected:
        const PixelFeatureExtractorBase<PixelType>* pixel_feature;
        const DistanceMetricBase<PixelType>* pixel_distance;
        const int s1;
        const int s2;
        const float t;
    };

    class TransitionPattern: public TransitionFeature<bool>{
    public:
        TransitionPattern(const PixelFeatureExtractorBase<PixelType>* pf_, const DistanceMetricBase<PixelType>* pd_,
                          const int s1_, const int s2_, const float theta_):
                TransitionFeature(pf_, pd_, s1_, s2_, theta_){}
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<FeatureType>& feat) const;

        virtual void computeFromPixelFeature(const VideoMat& pixelFeatures,
                                             std::vector<std::vector<FeatureType> >& feats) const;
    };

    class TransitionCounting: public TransitionFeature<float>{
    public:
        TransitionCounting(const PixelFeatureExtractorBase<PixelType>* pf_, const DistanceMetricBase<PixelType>* pd_,
                          const int s1_, const int s2_, const float theta_)
                : TransitionFeature(pf_, pd_, s1_, s2_, theta_){}
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<FeatureType>& feat) const;

        virtual void computeFromPixelFeature(const VideoMat& pixelFeatures,
                                             std::vector<std::vector<FeatureType> >& feats) const;
    };


}

#endif //DYNAMICSTEREO_PIXEL_FEATURE_H
