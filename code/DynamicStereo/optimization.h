//
// Created by yanhang on 3/3/16.
//

#ifndef DYNAMICSTEREO_OPTIMIZATION_H
#define DYNAMICSTEREO_OPTIMIZATION_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "base/depth.h"
#include "base/file_io.h"

namespace dynamic_stereo {
    class StereoOptimization {
    public:
        typedef int EnergyType;

        StereoOptimization(const FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                           const std::vector<EnergyType> &MRF_data_, const float MRFRatio_, const int nLabel_) :
                file_io(file_io_), kFrames(kFrames_), image(image_), MRF_data(MRF_data_), MRFRatio(MRFRatio_),
                nLabel(nLabel_), width(image_.cols), height(image_.rows) { }

        virtual void optimize(Depth &result, const int max_iter) const = 0;

        virtual double evaluateEnergy(const Depth &) const = 0;

    protected:
        const FileIO &file_io;
        const int kFrames;
        const cv::Mat &image;
        const std::vector<EnergyType> &MRF_data;
        const int nLabel;

        const float MRFRatio;

        const int width;
        const int height;
    };

    class FirstOrderOptimize : public StereoOptimization {
    public:
        FirstOrderOptimize(const FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                           const std::vector<EnergyType> &MRF_data_, const float MRFRatio_, const int nLabel_,
                           const EnergyType &weight_smooth_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

    private:
        void assignSmoothWeight();

        const EnergyType weight_smooth;
        std::vector<EnergyType> hCue;
        std::vector<EnergyType> vCue;
    };

    class SecondOrderOptimizeTRWS : public StereoOptimization {
    public:
        SecondOrderOptimizeTRWS(const FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                                const std::vector<EnergyType> &MRF_data_, const float MRFRatio_, const int nLabel_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

    private:
        EnergyType laml;
        EnergyType lamh;
        std::vector<int> refSeg;
    };

    class SecondOrderOptimizeFusionMove : public StereoOptimization {
    public:
        SecondOrderOptimizeFusionMove(const FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                                      const std::vector<EnergyType> &MRF_data_,
                                      const float MRFRatio_,
                                      const int nLabel_,
                                      const Depth &noisyDisp_,
                                      const double min_disp_, const double max_disp_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

    private:
        inline double lapE(const double x0, const double x1, const double x2) const{
            return std::min(std::abs(x0 + x2 - 2 * x1), trun);
        }

        void genProposal(std::vector<Depth> &proposals) const;

        void fusionMove(Depth &p1, const Depth &p2) const;

        const Depth &noisyDisp;
        const double min_disp;
        const double max_disp;
        const double trun;

        const int average_over;

        double laml;
        double lamh;
        std::vector<int> refSeg;
    };

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_OPTIMIZATION_H
