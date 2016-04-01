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
#include "model.h"

namespace dynamic_stereo {



    class StereoOptimization {
    public:
        typedef int EnergyType;

        StereoOptimization(const FileIO &file_io_, const int kFrames_, std::shared_ptr<StereoModel<EnergyType> > model_) :
                file_io(file_io_), kFrames(kFrames_), model(model_), width(model_->width), height(model_->height){ }

        virtual void optimize(Depth &result, const int max_iter) const = 0;

        virtual double evaluateEnergy(const Depth &) const = 0;

    protected:
        const FileIO &file_io;
        const int kFrames;
	    const int width;
	    const int height;
        std::shared_ptr<StereoModel<EnergyType> > model;
    };

    class FirstOrderOptimize : public StereoOptimization {
    public:
        FirstOrderOptimize(const FileIO &file_io_, const int kFrames_, std::shared_ptr<StereoModel<EnergyType> > model_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;
    };

    class SecondOrderOptimizeTRWS : public StereoOptimization {
    public:
        SecondOrderOptimizeTRWS(const FileIO &file_io_, const int kFrames_, std::shared_ptr<StereoModel<EnergyType> > model_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

    private:

        inline double lapE(const double x0, const double x1, const double x2) const{
            //return std::min((x0 + x2 - 2 * x1) * (x0 + x2 - 2 * x1), trun * trun);
            return std::min(std::abs(x0 + x2 - 2 * x1), trun);
        }
        double laml;
        double lamh;
        const double trun;
        std::vector<int> refSeg;
    };

    class SecondOrderOptimizeTRBP : public StereoOptimization {
    public:
        SecondOrderOptimizeTRBP(const FileIO &file_io_, const int kFrames_, std::shared_ptr<StereoModel<EnergyType> > model_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

    private:
        EnergyType laml;
        EnergyType lamh;
        std::vector<int> refSeg;
    };

    class SecondOrderOptimizeFusionMove : public StereoOptimization {
    public:
        SecondOrderOptimizeFusionMove(const FileIO &file_io_, const int kFrames_, std::shared_ptr<StereoModel<EnergyType> > model_,
                                      const Depth &noisyDisp_);

        virtual void optimize(Depth &result, const int max_iter) const;

        virtual double evaluateEnergy(const Depth &) const;

        const std::vector<int>& getRefSeg() const{
            return refSeg;
        }
    private:
        inline double lapE(const double x0, const double x1, const double x2) const{
            //return std::min((x0 + x2 - 2 * x1) * (x0 + x2 - 2 * x1), trun * trun);
            return std::min(std::abs(x0 + x2 - 2 * x1), trun);
        }

        void genProposal(std::vector<Depth> &proposals) const;

        void fusionMove(Depth &p1, const Depth &p2) const;

        const Depth &noisyDisp;
        const double trun;

        const int average_over;

        double laml;
        double lamh;
        std::vector<int> refSeg;
    };

    void toyTripleTRWS();

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_OPTIMIZATION_H
