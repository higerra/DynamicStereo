//
// Created by yanhang on 4/29/16.
//

#include "dynamicregularizer.h"
#include "../external/MRF2.2/GCoptimization.h"

namespace dynamic_stereo{

    void dynamicRegularization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output){
        vector<uchar *> inputPtr(input.size(), NULL);
        for(auto i=0; i<input.size(); ++i)


        auto processPixel = [&](int x, int y, int c){
            vector<double> MRF_data(input.size()*256, 0.0);
            for(auto i=0; i<input.size(); ++i){

            }
        };


    }

}//namespace dynamic_stereo