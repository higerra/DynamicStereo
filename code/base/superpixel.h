#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include "file_io.h"
#include "SLIC.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

namespace dynamic_rendering{

	class Frame;
	class Depth;

	struct SuperPixel{
		SuperPixel():center(Eigen::Vector2d(0,0)), average_confidence(0), label(0){}
		std::vector<int> indices;
		Eigen::Vector2d center;
		double average_confidence;
		int label;
	};


	class SuperPixelHelper{
	public:
		static void MatToImagebuffer(const cv::Mat&image,
		                             std::vector<unsigned int>&imagebuffer);

		static void ImagebufferToMat(const std::vector<unsigned int>&imagebuffer,
		                             const int imgwidth, const int imgheight,
		                             cv::Mat&image);

		static void computeSuperpixel(const FileIO& file_io,
		                              const int id,
		                              const Frame& frame,
		                              const double ratio, const int num,
		                              std::vector<int> &labels,
		                              int &num_labels);

		static bool loadSuperpixel(const FileIO& file_io,
		                           const int id,
		                           const int pixnum,
		                           std::vector<int> &labels,
		                           int &num_labels);

		static void gatherSuperpixel(const std::vector<Frame>&frames,
		                             const std::vector<std::vector<int> >&labels,
		                             const std::vector<int>&num_labels,
		                             const std::vector<Depth>& dynamic_confidence,
		                             std::vector<std::vector<SuperPixel> >& superpixel);

		static void pairSuperpixel(const std::vector<int>&labels,
		                           const int num_labels,
		                           int width, int height,
		                           std::vector<std::vector<int> >&pairmap);

		static void markSuperpixel(const SuperPixel& superpixel,
		                           cv::Mat& image,
		                           const double ratio = 1.0);
	};


} //namespace dynamic_rendering

#endif
