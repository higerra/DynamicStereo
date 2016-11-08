//
// Created by Yan Hang on 3/22/16.
//

#ifndef DYNAMICSTEREO_MODEL_H
#define DYNAMICSTEREO_MODEL_H

#include <opencv2/opencv.hpp>
#include <theia/theia.h>
#include <glog/logging.h>
#include <vector>
#include <string>
namespace dynamic_stereo{

	template<typename T>
	struct StereoModel{
		StereoModel(const cv::Mat& img_, const double downsample_, const int nLabel_, const double MRFRatio_, const double ws_):
				image(img_.clone()), downsample(downsample_), width(img_.cols), height(img_.rows), nLabel(nLabel_), MRFRatio(MRFRatio_), weight_smooth(ws_){
			CHECK(image.data) << "Empty image";
		}
		~StereoModel(){
		}

		void allocate(){
			unary.resize(width * height * nLabel);
			vCue.resize(width * height);
			hCue.resize(width * height);
		}

		std::vector<T> unary;
		std::vector<T> vCue;
		std::vector<T> hCue;
		const int nLabel;
		const int width;
		const int height;
		const double MRFRatio;

		//////////////////////////////////
		//NOTICE!!!! weigth_smooth is in original scale. Multiply with MRFRatio when needed
		const double weight_smooth;

		const cv::Mat image;
		const double downsample;

		double min_disp;
		double max_disp;

		inline const T operator()(int id, int label)const{
			CHECK_LT(id, width * height);
			CHECK_LT(label, nLabel);
			return unary[id * nLabel + label];
		}
		inline T& operator()(int id, int label){
			CHECK_LT(id, width * height);
			CHECK_LT(label, nLabel);
			return unary[id * nLabel + label];
		}

		inline double dispToDepth(const double d) const{
			return 1.0/(min_disp + d * (max_disp - min_disp) / (double) nLabel);
		}
		inline double depthToDisp(double depth) const{
			return (1.0 / depth * (double)nLabel - min_disp)/ (max_disp - min_disp);
		}
	};

    struct SfMModel{
        void init(const std::string& path);

        typedef std::pair<int, theia::ViewId> IdPair;
        theia::Reconstruction reconstruction;
        std::vector<IdPair> orderedId;

        inline const theia::Camera& getCamera(const int vid) const{
            CHECK_LT(vid, orderedId.size());
            return reconstruction.View(orderedId[vid].second)->Camera();
        }

        inline const theia::View* getView(const int vid) const{
            CHECK_LT(vid, orderedId.size());
            return reconstruction.View(orderedId[vid].second);
        }

        inline double warpPoint(const int vid1, const Eigen::Vector2d& pt1, const double depth, const int vid2, Eigen::Vector2d& imgpt2) const{
            CHECK_LT(vid1, orderedId.size());
            CHECK_LT(vid2, orderedId.size());
            const theia::Camera& cam1 = getCamera(vid1);
            const theia::Camera& cam2 = getCamera(vid2);
            Eigen::Vector3d spt = cam1.GetPosition() + cam1.PixelToUnitDepthRay(pt1) * depth;
            double depth2 = cam2.ProjectPoint(spt.homogeneous(), &imgpt2);
            return depth2;
        }
    };
}
#endif //DYNAMICSTEREO_MODEL_H
