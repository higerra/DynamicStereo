//
// Created by Yan Hang on 3/1/16.
//

#include "segment-image.h"

namespace segment_gb{
	//output: Mat with CV_32S type. Pixel values correspond to label Id
	//seg: grouped pixels. seg[i][j], j'th pixel in i'th segment
	int segment_image(const cv::Mat& input, cv::Mat& output, std::vector<std::vector<int> >& seg,
	                   const int smoothSize, float c, int min_size){
		CHECK(input.data != NULL);
		const int width = input.cols;
		const int height = input.rows;
		cv::Mat temp, smooth;
		input.convertTo(temp, cv::DataType<float>::type);
		cv::blur(temp, smooth, cv::Size(smoothSize,smoothSize));
		auto colorDiff = [](const cv::Mat& i1, const cv::Mat& i2, int x1, int y1, int x2, int y2){
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

		std::unique_ptr<universe> u(segment_graph(width * height, edges, c));

		// post process small components
		for (int i = 0; i < num; i++) {
			int a = u->find(edges[i].a);
			int b = u->find(edges[i].b);
			if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
				u->join(a, b);
		}

		output = cv::Mat(height, width, CV_32S, cv::Scalar::all(0));

		//remap labels
		std::vector<std::pair<int, int> > labelMap((size_t)width * height);
		int curMaxLabel = -1;
		int nLabel = -1;
		for (int i=0; i<width * height; ++i) {
			int comp = u->find(i);
			CHECK_LT(comp, width * height);
			labelMap[i] = std::pair<int,int>(comp, i);
		}
		std::sort(labelMap.begin(), labelMap.end());
		for(auto i=0; i<labelMap.size(); ++i){
			CHECK_GE(labelMap[i].first, 0);
			if(labelMap[i].first > curMaxLabel){
				curMaxLabel = labelMap[i].first;
				nLabel++;
			}
			int pixId = labelMap[i].second;
			output.at<int>(pixId/width, pixId%width) = nLabel;
		}

		nLabel++;
		seg.clear();
		seg.resize((size_t)(nLabel));
		for(int i=0; i<width * height; ++i) {
			seg[output.at<int>(i/width, i%width)].push_back(i);
		}
		u.reset();
		return nLabel;
	}

	cv::Mat visualizeSegmentation(const cv::Mat& input){
		CHECK(input.data);
		CHECK_EQ(input.type(), CV_32S);
		const int width = input.cols;
		const int height = input.rows;
		double minLabel, maxLabel;
		cv::minMaxLoc(input, &minLabel, &maxLabel);
		CHECK_LE(minLabel, std::numeric_limits<double>::epsilon());
		int nLabel = (int)maxLabel;
		CHECK_LE(maxLabel-(double)nLabel, std::numeric_limits<double>::epsilon());
		nLabel++;

		// pick random colors for each component
		std::vector<cv::Vec3b> colorTable(nLabel);

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, 255);

		for (int i = 0; i < nLabel; i++) {
			for(int j=0; j<3; ++j)
				colorTable[i][j] = (uchar)distribution(generator);
		}

		cv::Mat output(height, width, CV_8UC3, cv::Scalar::all(0));
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				const int label = input.at<int>(y,x);
				CHECK_LT(label, nLabel);
				output.at<cv::Vec3b>(y,x) = colorTable[label];
			}
		}
		return output;
	}

}//namespace segment_gb