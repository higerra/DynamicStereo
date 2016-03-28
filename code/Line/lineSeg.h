//
// Created by yanhang on 3/24/16.
//

#ifndef DYNAMICSTEREO_LINESEG_H
#define DYNAMICSTEREO_LINESEG_H

#include <theia/theia.h>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <glog/logging.h>
#include <string>
#include "../base/file_io.h"
#include "../base/utility.h"

namespace dynamic_stereo {

    struct Line{
        Line(): spt(0,0), ept(0,0){}
        Line(const Eigen::Vector2d& spt_, const Eigen::Vector2d& ept_, const cv::Mat& img):
                spt(spt_), ept(ept_){
            extractHist(img);
        };

        void extractHist(const cv::Mat& img);
        Eigen::Vector2d spt;
        Eigen::Vector2d ept;

        double length;

        //color histogram of left and right side
        std::vector<double> hL;
        std::vector<double> hR;

        static int sampleNum;
    };

    double compareLines(const Line& l1, const Line& l2);

    class LineSeg {
    public:
        LineSeg(const FileIO& file_io_, const int anchor_, const int tWindow);
        void undistort(const cv::Mat& input, cv::Mat& output, const theia::Camera& cam) const;
        void runLSD();
    private:
        typedef std::pair<int, theia::ViewId> IdPair;
        const FileIO& file_io;
        int width;
        int height;

        const int anchor;
        int offset;

        theia::Reconstruction reconstruction;
        std::vector<cv::Mat> images;
        std::vector<IdPair> orderedId;
    };
}

#endif //DYNAMICSTEREO_LINESEG_H
