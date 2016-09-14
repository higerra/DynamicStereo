//
// Created by yanhang on 1/8/16.
//

#ifndef QUADTRACKING_OPTICALFLOW_H
#define QUADTRACKING_OPTICALFLOW_H
#include "OpticalFlow/OpticalFlow.h"
#include "OpticalFlow/GaussianPyramid.h"
#include "file_io.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <string>
#include <list>
#include <glog/logging.h>
#include "utility.h"
#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>
#endif

namespace dynamic_stereo {
    class FileIO;

	struct FlowFrame{
		FlowFrame(){}
		FlowFrame(const cv::Mat& img_){
			init(img_);
		}
		FlowFrame(const int w, const int h){
			allocate(w,h);
		}
		FlowFrame(const DImage& img_){
			init(img_);
		}
		FlowFrame(const std::string& path){
			readFlowFile(path);
		}

		void init(const cv::Mat& img_);
		void init(const DImage& img_);
		inline int width() const {return w;}
		inline int height() const {return h;}
		inline Eigen::Vector2d getFlowAt(const Eigen::Vector2d& loc)const{
			return interpolation_util::bilinear<double, 2>(img.data(), width(),height(), loc);
		}
		inline Eigen::Vector2d getFlowAt(const int x, const int y) const{
			CHECK_LT(x, w);
			CHECK_LT(y, h);
			return getFlowAt(y*w+x);
		}
		inline Eigen::Vector2d getFlowAt(const int ind) const{
			CHECK_LT(2 * ind, img.size());
			return Eigen::Vector2d(img[2*ind], img[2*ind+1]);
		}
		inline void allocate(const int w_, const int h_){
			img.resize(w * h * 2, 0.0);
			w = w_;
			h = h_;
		}
		inline bool empty() const{
			return img.empty();
		}
		inline void clear(){
			img.clear();
		}
		void setValue(const int x, const int y, const int c, const double v);
		void setValue(const int x, const int y, const Eigen::Vector2d& v);
		inline std::vector<double>& data() {return img;}
		inline const std::vector<double>& data() const {return img;}


		bool readFlowFile(const std::string& path);
		void saveFlowFile(const std::string& path) const;
		inline bool isInsideFlowImage(const Eigen::Vector2d& loc) const{
			return loc[0] > 0 && loc[1] > 0 && loc[0] < w -1 && loc[1] < h - 1;
		}

	private:
		std::vector<double> img;
		int w, h;
	};

	class FlowEstimator{
	public:
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int downsample) = 0;
		virtual void downSample(const cv::Mat& input, cv::Mat& output, const int nLevel);
	};

	class FlowEstimatorCPU: public FlowEstimator{
	public:
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int nLevel);
	};

#ifdef USE_CUDA
	class FlowEstimatorGPU: public FlowEstimator{
	public:
		FlowEstimatorGPU();
		virtual void estimate(const cv::Mat& img1, const cv::Mat& img2, FlowFrame& flow, const int nLevel);
	private:
		cv::Ptr<cv::cuda::BroxOpticalFlow> brox;
	};
#endif

	namespace flow_util {
		void interpolateFlow(const FlowFrame &input, FlowFrame &output, const std::vector<bool> &mask, const bool fillHole = false);
		void warpImage(const cv::Mat &input, cv::Mat &output, const FlowFrame& flow);
		void visualizeFlow(const FlowFrame&, cv::Mat&);
		bool trackPoint(const Eigen::Vector2d& loc, const std::vector<FlowFrame>& flow, const int src, const int tgt, Eigen::Vector2d& res);
		void verifyFlow(const std::vector<FlowFrame>& flow_forward,
						const std::vector<FlowFrame>& flow_backward,
						const std::vector<cv::Mat>& frames,
						std::list<cv::Mat>& verifyimg,
						const int fid, Eigen::Vector2d loc);
		void resizeFlow(const FlowFrame& input, FlowFrame& output, const double ratio, const bool rescale = true);
		void resizeFlow(const FlowFrame& input, FlowFrame& output, const Eigen::Vector2i& dsize, const bool rescale = true);
		void computeMissingFlow(const FileIO& file_io, const int nlevel = 1);
	}

}//namespace dynamic_rendering

#endif //QUADTRACKING_OPTICALFLOW_H
