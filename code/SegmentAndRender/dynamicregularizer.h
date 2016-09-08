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
                            std::vector<cv::Mat> &output, double lambda = -1);

    void dumpOutSegment(const std::vector<cv::Mat>& images,
                        const std::vector<Eigen::Vector2d>& segment,
                        const std::string& path);
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_DYNAMICREGULARIZER_H
