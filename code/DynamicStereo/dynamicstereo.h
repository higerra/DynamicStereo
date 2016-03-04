//
// Created by yanhang on 2/24/16.
//

#ifndef DYNAMICSTEREO_DYNAMICSTEREO_H
#define DYNAMICSTEREO_DYNAMICSTEREO_H
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <theia/theia.h>
#include <time.h>
#include "base/utility.h"
#include "base/configurator.h"
#include "base/depth.h"
#include "base/file_io.h"
//#include "external/QPBO1.4/ELC.h"
#include "external/segment_ms/msImageProcessor.h"
#include "external/segment_gb/segment-image.h"
#include "external/MRF2.2/mrf.h"
//#include "MRF2.2/GCoptimization.h"
//#include <opengm/graphicalmodel/graphicalmodel.hxx>
//#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
//#include <opengm/functions/truncated_absolute_difference.hxx>

namespace dynamic_stereo {
    class DynamicStereo {
    public:
        DynamicStereo(const FileIO& file_io_, const int anchor_, const int tWindow_, const int downsample_, const double weight_smooth_,
                      const int dispResolution_ = 64, const double min_disp_ = -1, const double max_disp_ = -1);
        void verifyEpipolarGeometry(const int id1, const int id2,
                                                   const Eigen::Vector2d& pt,
                                                   cv::Mat &imgL, cv::Mat &imgR);
        void runStereo();

        inline int getAnchor()const{return anchor;}
        inline int gettWindow() const {return tWindow;}
        inline int getOffset() const {return offset;};
        inline int getDownsample() const {return downsample; }
	    void warpToAnchor(const Depth& refDisp, const std::string& prefix) const;
    private:
        typedef int EnergyType;
        //typedef opengm::GraphicalModel<EnergyType, opengm::Adder, opengm::ExplicitFunction<EnergyType>, opengm::SimpleDiscreteSpace<> > GraphicalModel;

        void initMRF();
        void computeMinMaxDisparity();
        void assignDataTerm();

        const FileIO& file_io;
        const int anchor;
        const int tWindow;
        const int downsample;
        int offset;

        int width;
        int height;

        const int dispResolution;
        const int pR; //radius of patch
        double min_disp;
        double max_disp;

        //downsampled version
        std::vector<cv::Mat> images;

	    //segmentation from meanshift, used for space varying CRF weight
	    std::vector<int> refSeg;

        theia::Reconstruction reconstruction;
        Depth dispUnary; //Noisy disparity map only based on unary term

        //for MRF
        std::vector<EnergyType> MRF_data;

        const double weight_smooth;
        const EnergyType MRFRatio;
        const double dispScale;
    };

    namespace MRF_util{
        void samplePatch(const cv::Mat& img, const Eigen::Vector2d& loc, const int pR, std::vector<double>& pix);
        double medianMatchingCost(const std::vector<std::vector<double> >& patches, const int refId);
    }

	namespace segment_uilt{
		void visualizeSegmentation(const std::vector<int>& labels, const int width, const int height, cv::Mat& output);
	}
}

#endif //DYNAMICSTEREO_DYNAMICSTEREO_H
