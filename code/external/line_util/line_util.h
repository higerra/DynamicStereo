//
// Created by yanhang on 5/24/16.
//

#ifndef DYNAMICSTEREO_LINEUTIL_H
#define DYNAMICSTEREO_LINEUTIL_H
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <limits>

namespace LineUtil{
    struct KeyLine{
    public:
        Eigen::Vector2d startPoint;
        Eigen::Vector2d endPoint;
        double lineLength;

        KeyLine():lineLength(0.0){}
        KeyLine(Eigen::Vector2d start_, Eigen::Vector2d end_):startPoint(start_), endPoint(end_){
            CHECK_GT((endPoint-startPoint).norm(), std::numeric_limits<double>::epsilon());
            lineLength = (endPoint - startPoint).norm();
        }

        inline cv::Point getStartPointCV() const{
            return cv::Point(startPoint[0], startPoint[1]);
        }

        inline cv::Point getEndPointCV() const{
            return cv::Point(endPoint[0], endPoint[1]);
        }

        inline Eigen::Vector2d getLineDir() const{
            CHECK_GT((endPoint-startPoint).norm(), std::numeric_limits<double>::epsilon());
            Eigen::Vector2d linedir = startPoint - endPoint;
            linedir.normalize();
            return linedir;
        }

        inline double getLength(){
            lineLength = (endPoint - startPoint).norm();
            return lineLength;
        }

        inline Eigen::Vector3d getHomo()const{
            CHECK_GT((endPoint-startPoint).norm(), std::numeric_limits<double>::epsilon());
            Eigen::Vector3d spt_homo = startPoint.homogeneous();
            Eigen::Vector3d ept_homo = endPoint.homogeneous();
            return spt_homo.cross(ept_homo);
        }
    };

    inline void dehomoPoint(Eigen::Vector3d& pt){
        if(std::abs(pt[2]) > std::numeric_limits<double>::epsilon()){
            pt[0] /= pt[2];
            pt[1] /= pt[2];
            pt[2] = 1.0;
        }
    }

    void detectLineSegments(const cv::Mat& input, std::vector<KeyLine>& output, const double min_length = 50);

    void vpDetection(const std::vector<KeyLine>& lines,
                     std::vector<std::vector<KeyLine> >& line_group,
                     std::vector<Eigen::Vector3d>& vp,
                     const int min_line_num,
                     const int max_cluster_num);

    void mergeLines(std::vector<KeyLine>& lines);

    void drawLines(cv::Mat& input, const std::vector<KeyLine>& lines,
                      const cv::Scalar c = cv::Scalar(0,0,255), const int thickness = 2);

    void drawLineGroups(cv::Mat& input, const std::vector<std::vector<KeyLine> >& lines);


}//namespace LineUtil


#endif //DYNAMICSTEREO_LINEUTIL_H
