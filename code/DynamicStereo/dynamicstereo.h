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
#include "../base/configurator.h"
#include "../base/depth.h"
#include "../base/file_io.h"
//#include "external/QPBO1.4/ELC.h"
#include "external/segment_ms/msImageProcessor.h"
#include "external/segment_gb/segment-image.h"
#include "external/MRF2.2/mrf.h"
#include "gridwarpping.h"
//#include "MRF2.2/GCoptimization.h"
//#include <opengm/graphicalmodel/graphicalmodel.hxx>
//#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
//#include <opengm/functions/truncated_absolute_difference.hxx>
#include "model.h"
namespace dynamic_stereo {
    class DynamicStereo {
    public:
        DynamicStereo(const FileIO& file_io_, const int anchor_, const int tWindow_, const int tWindowStereo_, const int downsample_, const double weight_smooth_,
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
		void disparityToDepth(const Depth& disp, Depth& depth);
		void bilateralFilter(const Depth& input, const cv::Mat& inputImg, Depth& output,
							 const int size, const double sigmas, const double sigmar, const double sigmau);

	    void getPatchArray(const double x, const double y, const int d, const int r, const theia::Camera& refCam, const int startid, const int endid, std::vector<std::vector<double> >& patches) const;

		//for debuging
		double dbtx;
		double dbty;
    private:
        typedef int EnergyType;

        void initMRF();
        void computeMinMaxDisparity();
        void assignDataTerm();
	    void assignSmoothWeight();

        const FileIO& file_io;
        const int anchor;
        const int tWindow;
		const int tWindowStereo;
        const int downsample;
        int offset;

		typedef std::pair<int, theia::ViewId> IdPair;
		std::vector<IdPair> orderedId;

	    std::shared_ptr<StereoModel<EnergyType> > model;

        int width;
        int height;

        const int dispResolution;
        const int pR; //radius of patch
        double min_disp;
        double max_disp;

        //downsampled version
        std::vector<cv::Mat> images;

        theia::Reconstruction reconstruction;
        Depth dispUnary; //Noisy disparity map only based on unary term
    };

	namespace utility{
		void visualizeSegmentation(const std::vector<int>& labels, const int width, const int height, cv::Mat& output);

		//test opencv built-in stereo matching
		void stereoSGBM(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& output);

		//depth, not dispartiy!
		void saveDepthAsPly(const std::string& path, const Depth& depth, const cv::Mat& image, const theia::Camera& cam, const int downsample);

	}
}

#endif //DYNAMICSTEREO_DYNAMICSTEREO_H
