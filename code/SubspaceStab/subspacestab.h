//
// Created by yanhang on 9/15/16.
//

#ifndef DYNAMICSTEREO_SUBSPACESTAB_H
#define DYNAMICSTEREO_SUBSPACESTAB_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace substab {

    struct SubSpaceStabOption {

        enum Direction {
            FORWARD = -1,
            BACKWARD = 0
        };

        SubSpaceStabOption(int tWindow_, int stride_, int smoothR_,
                           bool resize = false, bool crop = false, bool drawpoints = false,
                           int num_threads_ = 6, Direction dir_ = BACKWARD)
                : tWindow(tWindow_), stride(stride_), smoothR(smoothR_),
                  input_resize(resize), output_crop(crop), output_drawpoints(drawpoints), num_thread(num_threads_),
                  direction(dir_) {
            if (smoothR < 0)
                smoothR = tWindow / 2;
        }

        SubSpaceStabOption() : tWindow(30), stride(5), smoothR(15), input_resize(false),
                               output_crop(true), output_drawpoints(false), num_thread(6), direction(BACKWARD) {}

        int tWindow;
        int stride;
        int smoothR;
        bool input_resize;
        bool output_crop;
        bool output_drawpoints;
        int num_thread;
        Direction direction;
    };

    void cropImage(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output);

    void subSpaceStabilization(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                               const SubSpaceStabOption &option = SubSpaceStabOption());

}//namespace substab

#endif //DYNAMICSTEREO_SUBSPACESTAB_H
