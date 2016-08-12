//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_REGIONDESCRIPTOR_H
#define DYNAMICSTEREO_REGIONDESCRIPTOR_H

#include "mlutility.h"

namespace LineUtil{
	struct KeyLine;
}

namespace dynamic_stereo {
    namespace ML {
        //remove empty labels in video segments
        int compressSegments(std::vector<cv::Mat> &segments);

        //re-format video segments:
        int regroupSegments(const cv::Mat &segments,
                            std::vector<PixelGroup> &pixelGroup);

        void assignSegmentLabel(const std::vector<PixelGroup> &pixelGroup, const cv::Mat &mask,
                                std::vector<int> &label);

        void computeHoG(const std::vector<cv::Mat> &gradient, const PixelGroup &pixelIds,
                        std::vector<float> &hog, const int kBin);

        void extractFeature(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &gradient,
                            const cv::Mat &segments, const cv::Mat &mask, TrainSet &trainSet);

        //subroutine for computing feature bins
        void computeColor(const std::vector<cv::Mat> &colorImage, const PixelGroup &pg,
                          std::vector<float> &desc);

        void computeHoG(const std::vector<cv::Mat> &gradient, const PixelGroup &pixelIds,
                        std::vector<float> &hog, const int kBin);

        void computeShape(const PixelGroup &pg, const int width, const int height, std::vector<float> &desc);

        void computePosition(const PixelGroup &pg, const int width, const int height, std::vector<float> &desc);

        void computeLine(const PixelGroup &pg, const std::vector<std::vector<LineUtil::KeyLine> > &lineClusters,
                         std::vector<float> &desc);

        void computeTemporalPattern(const std::vector<cv::Mat>& colorImage, const PixelGroup& pg,
                                    std::vector<float>& desc, const int stride1 = 8, const int stride2 = 4);

        void visualizeSegmentLabel(const std::vector<cv::Mat> &images, const cv::Mat &segments,
                                   const std::vector<int> &label);

    }//namespace ML
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_REGIONDESCRIPTOR_H
