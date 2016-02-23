#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "camera.h"
#include "depth.h"
namespace dynamic_stereo{

    class FileIO;

    class Frame{
    public:
        Frame(const FileIO &file_io, int id);
        Frame(const cv::Mat& image_);
        Frame(){}

        void initialize(const FileIO& file_io, int id, bool nocamera = false);
	    void initialize(const int w, const int h){
		    image = cv::Mat(h, w, CV_8UC3, cv::Scalar(0,0,0));
		    width = w;
		    height = h;
	    }

        inline const int getWidth() const {return width;}
        inline const int getHeight() const {return height;}

        inline float getTime()const {return time;}
        inline void setTime(float timestamp){time = timestamp;}

        inline const bool isValidRGB(const Eigen::Vector2d& pt) const{
            return (pt[0]>=0 && pt[0]<width-1 && pt[1]>=0 && pt[1]<height-1);
        }
        inline const bool isValidDepth(const Eigen::Vector2d& pt)const {
            return (pt[0]>=0 && pt[0]<depth.getWidth()-1 && pt[1]>=0 && pt[1]<depth.getHeight()-1);
        }

        inline void blur(const int r, const double sigma_x = 0.0, const double sigma_y = 0.0){
            cv::GaussianBlur(image, image, cv::Size(r, r), sigma_x, sigma_y);
        }

        const Eigen::Vector3d getColor(Eigen::Vector2d pt, const bool downsample = false) const;
        const cv::Mat &getImage() const {return image;}
        const cv::Mat &getDownsampledImage() const{return downsampledImg;}

        inline const Eigen::Vector2d RGBToDepth(const Eigen::Vector2d& pt) const{
            return pt/depthratio;
        }
        inline const Eigen::Vector2d DepthToRGB(const Eigen::Vector2d& pt) const{
            return pt * depthratio;
        }

        inline const Depth& getDepth() const{return depth;}
        inline Depth& getDepth_nonConst(){return depth;}
        inline const Camera& getCamera()const{return camera;}
        inline Camera& getCamera_nonConst(){return camera;}

        bool isVisible(const Eigen::Vector3d& pt)const;

        static const double depthratio;
    private:
        int height;
        int width;
        float time;
        Depth depth;
        Camera camera;
        cv::Mat image;
        cv::Mat downsampledImg;
    };

    typedef std::vector<std::vector<Frame> > FramePyramid;

    void constructPyramid(const std::vector<Frame>& frames,
                          std::vector<std::vector<Frame> >& pyramid,
                          const int max_level);



}//namespace dynamic_rendering
#endif
