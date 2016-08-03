//
// Created by yanhang on 7/20/16.
//

#ifndef DYNAMICSTEREO_HOG3D_H
#define DYNAMICSTEREO_HOG3D_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include "descriptor.h"

namespace cv {

    class CV3DDescriptor: public Feature2D{
    public:
        virtual void prepareImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output) const = 0;
    };

    class CVHoG3D : public CV3DDescriptor {
    public:
        CVHoG3D(const int ss_, const int sr_, const int M_ = 4, const int N_ = 4, const int kSubBlock_ = 3);
        virtual void prepareImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output) const ;
        //use keypoint.octave as the z coordinate
        virtual void compute(InputArray image,
                             CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                             OutputArray descriptors);
    private:
        const int M; //number of cells in spatial dimension
        const int N; //number of cells in temporal dimension
        const int kSubBlock; //number of subblock inside each cell
        const int sigma_s; //spatial window size
        const int sigma_r; //temporal window size
    };

    class CVColor3D: public CV3DDescriptor{
    public:
        CVColor3D(const int ss_, const int sr_, const int M_ = 4, const int N_ = 4);
        virtual void prepareImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output) const;
        //use keypoint.octave as the z coordinate
        virtual void compute(InputArray image,
                             CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                             OutputArray descriptors);
    private:
        const int M; //number of cells in spatial dimension
        const int N; //number of cells in temporal dimension
        const int sigma_s; //spatial window size
        const int sigma_r; //temporal window size
    };

}//namespace cv
#endif //DYNAMICSTEREO_HOG3D_H
