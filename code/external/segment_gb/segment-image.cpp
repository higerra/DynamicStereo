//
// Created by Yan Hang on 3/1/16.
//

#include "segment-image.h"

namespace segment_gb{

	float TransitionPattern::compare(const std::vector<cv::Mat> &input, const int x1, const int y1, const int x2,
	                                const int y2) const {
		CHECK(!input.empty());
		CHECK_EQ(input[0].type(), CV_32FC3);
        std::vector<float> binDesc1;
        std::vector<float> binDesc2;
        binDesc1.reserve(input.size() * 2);
        binDesc2.reserve(input.size() * 2);
		for(auto v=0; v<input.size() - stride1; v+=stride1){
			double d1 = cv::norm(input[v].at<cv::Vec3f>(y1,x1) - input[v+stride1].at<cv::Vec3f>(y1,x1));
			double d2 = cv::norm(input[v].at<cv::Vec3f>(y2,x2) - input[v+stride1].at<cv::Vec3f>(y2,x2));
			if(d1 >= theta)
				binDesc1.push_back(1);
			else
				binDesc1.push_back(0);
			if(d2 >= theta)
				binDesc2.push_back(1);
			else
				binDesc2.push_back(0);
		}
		for(auto v=0; v<input.size() - stride2; v+=stride1/2) {
			double d1 = cv::norm(input[v].at<cv::Vec3f>(y1, x1) - input[v + stride2].at<cv::Vec3f>(y1, x1));
			double d2 = cv::norm(input[v].at<cv::Vec3f>(y2, x2) - input[v + stride2].at<cv::Vec3f>(y2, x2));
			if (d1 >= theta)
				binDesc1.push_back(1);
			else
				binDesc1.push_back(0);
			if (d2 >= theta)
				binDesc2.push_back(1);
			else
				binDesc2.push_back(0);
		}
		CHECK_EQ(binDesc1.size(), binDesc2.size());
		float diff_sum = 0.0f;
		for(auto i=0; i<binDesc1.size(); ++i){
			if(binDesc1[i] != binDesc2[i])
				diff_sum += 1.0f;
		}
		return diff_sum / (float)binDesc1.size();
	}

	float TransitionCounter::compare(const std::vector<cv::Mat> &input, const int x1, const int y1, const int x2,
	                                 const int y2) const {
		CHECK(!input.empty());
		CHECK_EQ(input[0].type(), CV_32FC3);

		float res = 0.0f;

		std::vector<float> binDesc1(2, 0.0f), binDesc2(2, 0.0f);
		float counter1 = 0.0f, counter2 = 0.0f;
		for(auto v=0; v<input.size() - stride1; v+=stride1){
			double d1 = cv::norm(input[v].at<cv::Vec3f>(y1,x1) - input[v+stride1].at<cv::Vec3f>(y1,x1));
			double d2 = cv::norm(input[v].at<cv::Vec3f>(y2,x2) - input[v+stride1].at<cv::Vec3f>(y2,x2));
			if(d1 >= theta)
				binDesc1[0] += 1.0;
			if(d2 >= theta)
				binDesc2[0] += 1.0;
			counter1 += 1.0;
		}

		res += std::abs(binDesc1[0] - binDesc2[0]) / counter1;

		for(auto v=0; v<input.size() - stride2; v+=stride1/2) {
			double d1 = cv::norm(input[v].at<cv::Vec3f>(y1, x1) - input[v + stride2].at<cv::Vec3f>(y1, x1));
			double d2 = cv::norm(input[v].at<cv::Vec3f>(y2, x2) - input[v + stride2].at<cv::Vec3f>(y2, x2));
			if (d1 >= theta)
				binDesc1[1] += 1.0;
			if (d2 >= theta)
				binDesc2[1] += 1.0;
			counter2 += 1.0;
		}

		res += std::abs(binDesc1[1] - binDesc2[1]) / counter2;
		return res / 2.0f;
	}

	void edgeAggregation(const std::vector<cv::Mat> &input, cv::Mat &output){
		CHECK(!input.empty());
		output.create(input[0].size(), CV_32FC1);
		output.setTo(cv::Scalar::all(0));
		for (auto i =0; i<input.size(); ++i) {
			cv::Mat edge_sobel(input[i].size(), CV_32FC1, cv::Scalar::all(0));
			cv::Mat gray, gx, gy;
			cvtColor(input[i], gray, CV_BGR2GRAY);
			cv::Sobel(gray, gx, CV_32F, 1, 0);
			cv::Sobel(gray, gy, CV_32F, 0, 1);
			for(auto y=0; y<gray.rows; ++y){
				for(auto x=0; x<gray.cols; ++x){
					float ix = gx.at<float>(y,x);
					float iy = gy.at<float>(y,x);
					edge_sobel.at<float>(y,x) = std::sqrt(ix*ix+iy*iy+FLT_EPSILON);
				}
			}
			output += edge_sobel;
		}

		double maxedge, minedge;
		cv::minMaxLoc(output, &minedge, &maxedge);
		if(maxedge > 0)
			output /= maxedge;
	}

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

	int segment_video(const std::vector<cv::Mat>& input, cv::Mat& output,
	                  const int smoothSize, const float c, const float theta, const int min_size){
		CHECK(!input.empty());
		const int width = input[0].cols;
		const int height = input[0].rows;

		printf("preprocessing\n");
		std::vector<cv::Mat> smoothed(input.size());
		for(auto v=0; v<input.size(); ++v){
			cv::Mat temp;
			input[v].convertTo(temp, cv::DataType<float>::type);
			cv::blur(temp, smoothed[v], cv::Size(smoothSize, smoothSize));
		}

		cv::Mat edgeMap;
		edgeAggregation(smoothed, edgeMap);

		const int stride1 = 8;
		const int stride2 = (int)input.size() / 2;

		std::shared_ptr<TemporalComparator> comparator(new TransitionPattern(stride1, stride2, theta));

		// build graph
		std::vector<edge> edges((size_t)width*height*4);
		int num = 0;

		printf("Computing edge weight\n");
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float edgeness = edgeMap.at<float>(y,x);
				if (x < width - 1) {
					edges[num].a = y * width + x;
					edges[num].b = y * width + (x + 1);
					edges[num].w = comparator->compare(smoothed, x, y, x+1, y) * edgeness;
					num++;
				}

				if (y < height - 1) {
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + x;
					edges[num].w = comparator->compare(smoothed, x, y, x, y+1) * edgeness;
					num++;
				}

				if ((x < width - 1) && (y < height - 1)) {
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + (x + 1);
					edges[num].w = comparator->compare(smoothed, x, y, x+1, y+1) * edgeness;
					num++;
				}

				if ((x < width - 1) && (y > 0)) {
					edges[num].a = y * width + x;
					edges[num].b = (y - 1) * width + (x + 1);
					edges[num].w = comparator->compare(smoothed, x, y, x+1, y-1) * edgeness;
					num++;
				}
			}
		}

		printf("segment graph\n");

		std::unique_ptr<universe> u(segment_graph(width * height, edges, c));

		printf("post processing\n");
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