//
// Created by yanhang on 11/1/16.
//

#ifndef DYNAMICSTEREO_CINEMAGRAPH_H
#define DYNAMICSTEREO_CINEMAGRAPH_H

#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

namespace dynamic_stereo {
    namespace Cinemagraph {

        struct Cinemagraph {
            cv::Mat background;
            cv::Mat blend_map;

            std::vector<cv::Mat> pixel_mat_display;
            std::vector<cv::Mat> pixel_mat_flashy;
            std::vector<std::vector<Eigen::Vector2i> > pixel_loc_display;
            std::vector<std::vector<Eigen::Vector2i> > pixel_loc_flashy;
            std::vector<Eigen::Vector2i> ranges_display;
            std::vector<Eigen::Vector2i> ranges_flashy;
        };

        void CreatePixelMat(const std::vector<cv::Mat>& warped, const std::vector<Eigen::Vector2i>& segment,
                            const Eigen::Vector2i& range, cv::Mat& output);

        void ComputeBlendMap(const std::vector<std::vector<Eigen::Vector2i> >& segments,
                             const int width, const int height, const int blend_R, cv::Mat& blend_map);

        void RenderCinemagraph(const Cinemagraph &cinemagraph, std::vector<cv::Mat> &output, const int kOutputFrame,
                               const bool start_from_reference = true);

        bool ReadCinemagraph(const std::string &path);

        void SaveCinemagraph(const std::string &path, Cinemagraph &output);
    }
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_CINEMAGRAPH_H
