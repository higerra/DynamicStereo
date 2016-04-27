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
#include "../base/utility.h"
#include "../base/depth.h"
#include "../base/file_io.h"

#include "dynamic_utility.h"

namespace dynamic_stereo {
    class DynamicStereo {
    public:
	    typedef double EnergyType;

        DynamicStereo(const FileIO& file_io_, const int anchor_, const int tWindow_, const int tWindowStereo_, const int downsample_, const double weight_smooth_,
                      const int dispResolution_ = 64, const double min_disp_ = -1, const double max_disp_ = -1);

        void runStereo(const cv::Mat& inputMask, Depth& result, cv::Mat& mask);

        inline int getAnchor()const{return anchor;}
        inline int gettWindow() const {return tWindow;}
        inline int getOffset() const {return offset;}
        inline int getDownsample() const {return downsample; }
		inline const SfMModel& getSfMModel() const{return sfmModel;}

		void warpToAnchor(const Depth& refDisp, const cv::Mat& mask, const int startid, const int endid, std::vector<cv::Mat>& warpped) const;

		void disparityToDepth(const Depth& disp, Depth& depth);
		void bilateralFilter(const Depth& input, const cv::Mat& inputImg, Depth& output,
							 const int size, const double sigmas, const double sigmar, const double sigmau);

	    void getPatchArray(const double x, const double y, const int d, const int r, const theia::Camera& refCam, const int startid, const int endid, std::vector<std::vector<double> >& patches) const;
	    const std::shared_ptr<StereoModel<EnergyType> > getMRFModel() const{
		    return model;
	    }
		//for debuging
		double dbtx;
		double dbty;
    private:
        void initMRF();

        void assignDataTerm();
	    void assignSmoothWeight();
		void computeFrequencyConfidence(const double alpha = 3, const double beta = 2.0);

	    //analysis the
	    double getFrequencyConfidence(const int fid, const int x, const int y, const int d) const;

        const FileIO& file_io;
        const int anchor;
        const int tWindow;
		const int tWindowStereo;
        const int downsample;
        int offset;

	    std::shared_ptr<StereoModel<EnergyType> > model;

        int width;
        int height;

        const int dispResolution;
        const int pR; //radius of patch

        //downsampled version
        std::vector<cv::Mat> images;

		SfMModel sfmModel;
        Depth dispUnary; //Noisy disparity map only based on unary term
    };
}

#endif //DYNAMICSTEREO_DYNAMICSTEREO_H
