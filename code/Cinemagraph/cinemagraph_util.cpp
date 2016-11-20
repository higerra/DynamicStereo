//
// Created by yanhang on 11/19/16.
//

#include "cinemagraph_util.h"

#include <ceres/ceres.h>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    namespace Cinemagraph{

        struct AreaRatioFunctor{
        public:
            AreaRatioFunctor(const std::vector<Eigen::Vector2i>& locs, const double weight):
                    locs_(locs), weight_(weight){}

            bool operator() (const double * const x1, const double * const y1,
                             const double * const x2, const double * const y2,
                             const double * const x3, const double * const y3,
                             const double * const x4, const double * const y4, double* residual) const{

                return true;
            }
        private:
            const std::vector<Eigen::Vector2i>& locs_;
            const double weight_;
        };

        static void RefineQuadNonLinear(const std::vector<Eigen::Vector2i>& locs, std::vector<cv::Point>& corners){

        }

        void ApproximateQuad(const std::vector<Eigen::Vector2i>& locs, const int width, const int height,
                             std::vector<int>& output, const bool refine){
            if(output.empty()){
                output.resize(8, -1);
            }
            const int min_size = 2000;
            const float max_aspect_ratio = 5;
            const float min_overlap_ratio = 0.90;
            printf("============================\n");
            printf("Size: %d\n", (int)locs.size());
            if(locs.size() < min_size){
                return;
            }
            Mat mask(height, width, CV_8UC1, Scalar::all(0));
            vector<vector<cv::Point> > contours;
            for(const auto& pt: locs){
                mask.at<uchar>(pt[1], pt[0]) = (uchar)255;
            }
            cv::findContours(mask, contours, cv::noArray(), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            vector<cv::Point> approx_contour;
            double approx_epsilon = 1.0;
            CHECK(!contours.empty());
            while(approx_epsilon < 50) {
                cv::approxPolyDP(contours[0], approx_contour, approx_epsilon, true);
                if (approx_contour.size() == 4) {
                    break;
                }
                approx_epsilon += 0.1;
            }
            if(approx_contour.size() != 4){
                return;
            }
            cv::RotatedRect min_rect = cv::minAreaRect(approx_contour);
            const int min_edge = std::min(min_rect.size.width, min_rect.size.height);
            const int max_edge = std::max(min_rect.size.width, min_rect.size.height);
            const float ar = (float)max_edge / (float)min_edge;
            printf("ar: %.3f\n", ar);
            if(ar > max_aspect_ratio){
                return;
            }

            if(refine){

            }
            //check the overlap region
            int overlap_count = 0;
            for(const auto& pt: locs){
                if(cv::pointPolygonTest(approx_contour, cv::Point2f(pt[0], pt[1]), false) > 0){
                    overlap_count++;
                }
            }
            const float overlap_ratio  = (float)overlap_count / (float)locs.size();
            printf("Epsilon: %.3f, overlap: %.3f\n", approx_epsilon, overlap_ratio);
            Mat img_contour(mask.size(), CV_8UC3, Scalar::all(0));
            for(const auto& pt: locs){
                img_contour.at<Vec3b>(pt[1], pt[0]) = Vec3b(255,0,0);
            }
            vector<vector<cv::Point> > contour_vis{approx_contour};
            cv::drawContours(img_contour, contour_vis, 0, Scalar(0,0,255), 2);
            imshow("contour", img_contour);

            for(const auto& pt: approx_contour){
                cout << pt.x << ' ' << pt.y << endl;
            }
            waitKey(0);
            if(overlap_ratio < min_overlap_ratio){
                return;
            }

            output.clear();
            for(const auto& pt: approx_contour){
                output.push_back(pt.x);
                output.push_back(pt.y);
            }
        }

    }//namespace Cinemagraph
}//namespace dynamic_stereo