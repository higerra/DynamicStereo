//
// Created by yanhang on 3/28/16.
//

#ifndef DYNAMICSTEREO_GRIDWARPPING_H
#define DYNAMICSTEREO_GRIDWARPPING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <theia/theia.h>
#include <ceres/ceres.h>
#include <memory>
#include <vector>
#include <list>
#include <iostream>
#include "model.h"
#include "../base/file_io.h"
#include "../base/depth.h"
#include "../base/utility.h"

namespace dynamic_stereo {
    class GridWarpping {
    public:
        typedef int EnergyType;
        typedef std::vector<std::pair<int, theia::TrackId> > OrderedIDSet;

        GridWarpping(const FileIO &file_io_, const int anchor_, const std::vector<cv::Mat> &images_,
                     const StereoModel<EnergyType> &model_,
                     const theia::Reconstruction &reconstruction_, const OrderedIDSet &orderedId_, Depth &refDepth_,
                     const int downsample_, const int offset_, const int gw = 32, const int gh = 18);


        void getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind, Eigen::Vector4d &w) const;

        void computePointCorrespondence(const int id, std::vector<Eigen::Vector2d> &refPt,
                                        std::vector<Eigen::Vector2d> &srcPt) const;

	    //wf: output dense warping field
	    void computeWarppingField(const std::vector<Eigen::Vector2d>& refPt,
	                              const std::vector<Eigen::Vector2d>& srcPt,
	                              std::vector<std::vector<Eigen::Vector2d> >& wf, cv::Mat& vis) const;

    private:
        const FileIO &file_io;
        const std::vector<cv::Mat> &images;
        const StereoModel<EnergyType> &model;
        const theia::Reconstruction &reconstruction;
        const OrderedIDSet &orderedId;
        const Depth &refDepth;
        const int anchor;

        const int downsample;
        const int offset;

        std::vector<Eigen::Vector2d> gridLoc;

        int width;
        int height;
        int gridW;
        int gridH;
        double blockW;
        double blockH;
    };
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_GRIDWARPPING_H
