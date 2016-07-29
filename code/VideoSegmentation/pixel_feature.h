//
// Created by yanhang on 7/29/16.
//

#ifndef DYNAMICSTEREO_PIXEL_FEATURE_H
#define DYNAMICSTEREO_PIXEL_FEATURE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dynamic_stereo{
    using VideoMat = std::vector<cv::Mat>;

    template<typename T>
    class DistanceMetricBase;

    /////////////////////////////////////////////////////////////
    //pixel level feature
    template<typename T>
    class PixelFeatureExtractorBase{
    public:
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<T>& feat) const = 0;
        void extractImage(const cv::Mat& input, std::vector<std::vector<T> >& feats) const{
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
    };

    class PixelValue: public PixelFeatureExtractorBase<float>{
    public:
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<float>& feat) const;
    };

    class BRIEFWrapper: public PixelFeatureExtractorBase<bool>{
        virtual void extractPixel(const cv::Mat& input, const int x, const int y, std::vector<bool>& feat) const;
    };

    ////////////////////////////////////////////////////////////
    //temporal feature
    template<typename T>
    class TemporalFeatureExtractorBase{
    public:
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<float>& feat) const = 0;
        void extractVideo(const VideoMat& input, std::vector<std::vector<T> >& feats) const{
            CHECK(!input.empty());
            const int width = input[0].cols;
            const int height = input[0].rows;
            feats.resize((size_t)(width * height));

            for(auto y=0; y<height; ++y){
                for(auto x=0; x<width; ++x){
                    extractPixel(input, x,y, feats[y*width+x]);
                }
            }
        }
    };

    template<typename T>
    class TransitionFeature: public TemporalFeatureExtractorBase<T>{
    public:
        TransitionFeature(const PixelFeatureExtractorBase* pf_, const DistanceMetricBase<T>* pd_,
                          const int s1_, const int s2_, const float theta_):
                pixel_feature(pf_), pixel_distance(pd_), s1(s1_), s2(s2_), t(theta_){
            CHECK_GT(stride1(), 0);
            CHECK(pixel_feature);
            CHECK(pixel_distance);
        }
        inline int stride1() const {return s1;}
        inline int stride2() const {return s2;}
        inline float theta() const {return t;}

        inline const PixelFeatureExtractorBase* getPixelFeatureExtractor() const{
            return pixel_feature;
        }
        inline const DistanceMetricBase<float>* getPixelFeatureComparator() const{
            return pixel_distance;
        }
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<float>& feat) const = 0;
    protected:
        const PixelFeatureExtractorBase* pixel_feature;
        const DistanceMetricBase<T>* pixel_distance;
        const int s1;
        const int s2;
        const float t;
    };

    class TransitionPattern: public TransitionFeature<bool>{
    public:
        TransitionPattern(const PixelFeatureExtractorBase* pf_, const DistanceMetricBase<bool>* pd_,
                          const int s1_, const int s2_, const float theta_):
                TransitionFeature(pf_, pd_, s1_, s2_, theta_){}
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<bool>& feat) const;
    };

    class TransitionCounting: public TransitionFeature<float>{
    public:
        TransitionCounting(const PixelFeatureExtractorBase* pf_, const DistanceMetricBase<float>* pd_,
                          const int s1_, const int s2_, const float theta_)
                : TransitionFeature(pf_, pd_, s1_, s2_, theta_){}
        virtual void extractPixel(const VideoMat& input, const int x, const int y, std::vector<float>& feat) const;
    };


}

#endif //DYNAMICSTEREO_PIXEL_FEATURE_H
