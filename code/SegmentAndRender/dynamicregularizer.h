//
// Created by yanhang on 4/29/16.
//

#ifndef DYNAMICSTEREO_DYNAMICREGULARIZER_H
#define DYNAMICSTEREO_DYNAMICREGULARIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <glog/logging.h>
#include <Eigen/Eigen>

namespace dynamic_stereo {

    class Depth;

    float GetRPCAWeight(const cv::Mat& pixels, const int stride = 4, const int theta = 30);

    void getSegmentRange(const std::vector<cv::Mat>& visMaps,
                         const std::vector<std::vector<Eigen::Vector2i> >& segments,
                         std::vector<Eigen::Vector2i>& ranges);

    void SearchFlashyLoop(cv::Mat& pixel_mat, Eigen::Vector2i& range);

    void filterShortSegments(std::vector<std::vector<Eigen::Vector2i> >& segments,
                             std::vector<Eigen::Vector2i>& ranges,
                             const int minFrame);

    void renderToMask(const std::vector<cv::Mat>& input,
                      const std::vector<std::vector<Eigen::Vector2i> >& segments,
                      const std::vector<Eigen::Vector2i>& ranges, std::vector<cv::Mat>& output);

    void regularizationAnisotropic(const std::vector<cv::Mat> &input,
                                   const std::vector<std::vector<Eigen::Vector2i> > &segments,
                                   std::vector<cv::Mat> &output, const double weight_smooth);

    void regularizationPoisson(const std::vector<cv::Mat> &input,
                               const std::vector<std::vector<Eigen::Vector2i> > &segments,
                               std::vector<cv::Mat> &output, const double ws, const double wt);

    void temporalMedianFilter(const std::vector<cv::Mat>& input,
                              const std::vector<std::vector<Eigen::Vector2i> >& segments,
                              std::vector<cv::Mat>& output, const int r);

    void regularizationFlashy(const std::vector<cv::Mat> &input,
                              const std::vector<std::vector<Eigen::Vector2i> > &segments,
                              std::vector<cv::Mat> &output);

    void regularizationRPCA(const std::vector<cv::Mat> &input,
                            const std::vector<std::vector<Eigen::Vector2i> > &segments,
                            const std::vector<float>& lambdas,
                            std::vector<cv::Mat> &output);

    void dumpOutSegment(const std::vector<cv::Mat>& images,
                        const std::vector<Eigen::Vector2d>& segment,
                        const std::string& path);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DYNAMICREGULARIZER_H
