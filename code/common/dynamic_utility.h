//
// Created by yanhang on 4/10/16.
//

#ifndef DYNAMICSTEREO_DYNAMIC_UTILITY_H
#define DYNAMICSTEREO_DYNAMIC_UTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <memory>

namespace theia{
	class Camera;
}

namespace dynamic_stereo{
    struct SfMModel;
    class Depth;
    class FileIO;

    namespace utility{
        void visualizeSegmentation(const std::vector<int>& labels, const int width, const int height, cv::Mat& output);

        //depth, not dispartiy!
        void saveDepthAsPly(const std::string& path, const Depth& depth, const cv::Mat& image, const theia::Camera& cam, const int downsample);

        void verifyEpipolarGeometry(const FileIO& file_io,
                                    const dynamic_stereo::SfMModel& sfm,
                                    const int id1, const int id2,
                                    const Eigen::Vector2d& pt,
                                    cv::Mat &imgL, cv::Mat &imgR);
        void computeMinMaxDepth(const dynamic_stereo::SfMModel& sfm, const int refId, double& min_depth, double& max_depth);



        double getFrequencyScore(const cv::Mat& colorArray, const int min_frq);
    }//namespace utility
}//namespace dynamic_stereo
#endif //DYNAMICSTEREO_DYNAMIC_UTILITY_H
