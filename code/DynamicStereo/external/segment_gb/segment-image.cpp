//
// Created by Yan Hang on 3/1/16.
//

#include "segment-image.h"
#include <memory>
#include <random>

namespace segment_gb{

	void makeMask(float sigma, std::vector<float>& mask){
		sigma = std::max(sigma, 0.01F);
		int len = (int)ceil(sigma * 4.0) + 1;
		mask.resize(len);
		for(auto i=0; i<len; ++i)
			mask[i] = (float)std::exp(-0.5 * (i*i/(sigma*sigma)));
		float sum = 0;
		for(int i=1; i<len; ++i)
			sum += fabs(mask[i]);
		sum = 2.0f * sum + fabs(mask[0]);
		CHECK_GT(sum, 0);
		for(int i=0; i<len; ++i)
			mask[i] /= sum;
	}

	void convolveMat(const cv::Mat& input, cv::Mat& output, const std::vector<float>& mask){
		cv::Mat fmat;
		input.convertTo(fmat, CV_32F);
		const int width = input.cols;
		const int height = input.rows;
		output = cv::Mat(height, width, CV_32FC3);
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				cv::Vec3f sum = mask[0] * fmat.at<cv::Vec3f>(y,x);
				for(int i=1; i<mask.size(); ++i)
					sum += mask[i] * (fmat.at<cv::Vec3f>(y, std::max(x-i, 0)) + fmat.at<cv::Vec3f>(y, std::min(x+i, width-1)));
				output.at<cv::Vec3f>(y,x) = sum;
			}
		}
	}

	int segment_image(const cv::Mat& input, cv::Mat& output, std::vector<std::vector<int> >& seg,
	                   float sigma, float c, int min_size){
		CHECK(input.data != NULL);
		const int width = input.cols;
		const int height = input.rows;
		cv::Mat temp, smooth;

		std::vector<float> mask;
		makeMask(sigma, mask);

		convolveMat(input, temp, mask);
		convolveMat(temp, smooth, mask);

		auto colorDiff = [](const cv::Mat& i1, const cv::Mat& i2, int x1, int y1, int x2, int y2){
			CHECK_EQ(i1.size(), i2.size());
			CHECK(x1 >= 0 && x1 <i1.cols);
			CHECK(y1 >= 0 && y1 <i1.rows);
			CHECK(x2 >= 0 && x2 <i2.cols);
			CHECK(y2 >= 0 && y2 <i2.rows);

			cv::Vec3f c1 = i1.at<cv::Vec3f>(y1, x1);
			cv::Vec3f c2 = i2.at<cv::Vec3f>(y2, x2);
			return std::sqrt(((double)c1[0] - (double)c2[0]) * ((double)c1[0] - (double)c2[0]) +
			                 ((double)c1[1] - (double)c2[1]) * ((double)c1[1] - (double)c2[1]) +
			                 ((double)c1[2] - (double)c2[2]) * ((double)c1[2] - (double)c2[2]));
		};
		// build graph
		std::vector<edge> edges((size_t)width*height*4);
		int num = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (x < width - 1) {
					edges[num].a = y * width + x;
					edges[num].b = y * width + (x + 1);
					edges[num].w = (float)colorDiff(smooth, smooth, x, y, x+1, y);
					num++;
				}

				if (y < height - 1) {
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + x;
					edges[num].w = (float)colorDiff(smooth, smooth, x, y, x, y+1);
					num++;
				}

				if ((x < width - 1) && (y < height - 1)) {
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + (x + 1);
					edges[num].w = (float)colorDiff(smooth, smooth, x, y, x+1, y+1);
					num++;
				}

				if ((x < width - 1) && (y > 0)) {
					edges[num].a = y * width + x;
					edges[num].b = (y - 1) * width + (x + 1);
					edges[num].w = (float)colorDiff(smooth, smooth, x, y, x+1, y-1);
					num++;
				}
			}
		}

		std::unique_ptr<universe> u(segment_graph(width * height, num, edges.data(), c));

		// post process small components
		for (int i = 0; i < num; i++) {
			int a = u->find(edges[i].a);
			int b = u->find(edges[i].b);
			if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
				u->join(a, b);
		}

		output = cv::Mat(height, width, CV_8UC3);

// pick random colors for each component
		std::vector<cv::Vec3b> colorTable((size_t)width * height);


		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, 255);

		for (int i = 0; i < width * height; i++) {
			for(int j=0; j<3; ++j)
				colorTable[i][j] = (uchar)distribution(generator);
		}

		int nLabels = -1;
		std::vector<int> labels((size_t)width * height);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int comp = u->find(y * width + x);
				labels[y*width+x] = comp;
				nLabels = std::max(nLabels, comp);
				output.at<cv::Vec3b>(y,x) = colorTable[comp];
			}
		}
		seg.clear();

		seg.resize((size_t)(nLabels+1));
		for(int i=0; i<width * height; ++i) {
			seg[labels[i]].push_back(i);
		}
		u.reset();
		return nLabels;
	}
}