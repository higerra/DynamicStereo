//
// Created by yanhang on 9/15/16.
//

#ifndef DYNAMICSTEREO_SUBSPACESTAB_H
#define DYNAMICSTEREO_SUBSPACESTAB_H

#include <vector>

namespace cv{
    class Mat;
}
namespace substab {

    struct SubSpaceStabOption {
        SubSpaceStabOption(int tWindow_ = 30, int stride_ = 5, int smoothR_ = -1,
                           bool resize = false, bool crop = false, bool drawpoints = false,
                           int num_threads_ = 6)
                : tWindow(tWindow_), stride(stride_), smoothR(smoothR_),
                  input_resize(resize) ,output_crop(crop), output_drawpoints(drawpoints), num_thread(num_threads_){
            if(smoothR < 0)
                smoothR = tWindow / 2;
        }

        int tWindow;
        int stride;
        int smoothR;
        bool input_resize;
        bool output_crop;
        bool output_drawpoints;
        int num_thread;
    };

    void cropImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output);

    void subSpaceStabilization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output,
                               const SubSpaceStabOption& option = SubSpaceStabOption());

}//namespace substab

#endif //DYNAMICSTEREO_SUBSPACESTAB_H
