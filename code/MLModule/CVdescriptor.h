//
// Created by yanhang on 7/20/16.
//

#ifndef DYNAMICSTEREO_HOG3D_H
#define DYNAMICSTEREO_HOG3D_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace dynamic_stereo{
    namespace ML {
        class Feature3D {
        public:
            virtual void constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const = 0;

            int getDim() { return dim; }

        protected:
            int dim;
        };



        class HoG3D : public Feature3D {
        public:
            HoG3D(const int M_ = 4, const int N_ = 4, const int kSubBlock_ = 3);

            //Note: input image should be gradient of 3 channels: gx, gy, gz, in float type
            virtual void constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const;

        private:
            Eigen::MatrixXf P;
            const int kSubBlock;
            const int M;
            const int N;
        };

        class Color3D : public Feature3D {
        public:
            Color3D(const int M_ = 4, const int N_ = 4) : M(4), N(4) {}

            virtual void constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const;

        private:
            const int M;
            const int N;
        };
    }//namespace ML
}//namespace dynamic_stereo

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
