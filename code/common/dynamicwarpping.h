//
// Created by yanhang on 4/10/16.
//

#ifndef DYNAMICSTEREO_DYNAMICWARPPING_H
#define DYNAMICSTEREO_DYNAMICWARPPING_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

namespace theia{
	class Camera;
}

namespace dynamic_stereo {
	struct SfMModel;
	class Depth;
	class FileIO;

    class DynamicWarpping {
    public:
        DynamicWarpping(const FileIO& file_io_, const int anchor_, const int tWindow_, const int nLabel_,
						const double wSmooth = -1);

        void warpToAnchor(const std::vector<cv::Mat>& images,
						  const std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay,
						  const std::vector<std::vector<Eigen::Vector2d> >& segmentsFlashy,
						  std::vector<cv::Mat>& output, const int kFrame) const;
		void preWarping(const cv::Mat& mask, std::vector<cv::Mat>& warped) const;

        inline int getOffset() const{return offset;}
		const Depth& getDepth() const{return *refDepth;}
		const std::shared_ptr<Depth> mutableDepth(){return refDepth;}

	    inline double depthToDisp(const double d, const double min_depth, const double max_depth) const{
		    CHECK_GT(min_depth, 0.0);
		    CHECK_GT(max_depth, 0.0);
		    double min_disp = 1.0 / max_depth;
		    double max_disp = 1.0 / min_depth;
		    return (1.0 / d * (double)nLabel - min_disp)/ (max_disp - min_disp);
	    }
    private:
        void initZBuffer();
        void updateZBuffer(const Depth* depth, Depth* zb, const theia::Camera& cam1, const theia::Camera& cam2) const;
        const FileIO& file_io;
        std::shared_ptr<SfMModel> sfmModel;
        std::shared_ptr<Depth> refDepth;
        const int anchor;
	    const int nLabel;
		const int tWindow;
		double downsample;
        int offset;
        std::vector<std::shared_ptr<Depth> > zBuffers;
	    std::vector<double> min_depths;
	    std::vector<double> max_depths;
    };
}

#endif //DYNAMICSTEREO_DYNAMICWARPPING_H
