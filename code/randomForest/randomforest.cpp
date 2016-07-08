//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	//feature based over segments
	//3 channels of mean L*a*b
	//24 channels of histogram of color changes.
	//3 channels of shape: mean area,
	void extractFeature(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& segments, const cv::Mat& mask,
	                    const FeatureOption& option, TrainSet& trainSet){
		CHECK(!images.empty());
		CHECK(!segments.empty());
		CHECK_EQ(images.size(), segments.size());
		CHECK_EQ(images[0].size(), segments[0].size());

		vector<vector<vector<int> > > pixelGroup;
		vector<vector<int> > regionSpan;
		printf("Regrouping...\n");
		const int kSeg = regroupSegments(segments, pixelGroup, regionSpan);

		const int tSeg = 100;
		visualizeSegmentGroup(images, pixelGroup[tSeg], regionSpan[tSeg]);
	}

	void compressSegments(std::vector<cv::Mat>& segments){
		
	}

	int regroupSegments(const std::vector<cv::Mat> &segments,
	                     std::vector<std::vector<std::vector<int> > > &pixelGroup,
	                     std::vector<std::vector<int> > &regionSpan){

		CHECK(!segments.empty());
		CHECK_EQ(segments[0].type(), CV_32S);

		const int width = segments[0].cols;
		const int height = segments[0].rows;

		int kSeg = 0;
		for(const auto& seg: segments){
			double minid, maxid;
			cv::minMaxLoc(seg, &minid, &maxid);
			kSeg = std::max(kSeg, (int)maxid);
		}
		kSeg++;

		printf("Number of segments: %d\n", kSeg);

		pixelGroup.resize((size_t)kSeg);
		regionSpan.resize((size_t)kSeg);

		for(auto& pg: pixelGroup)
			pg.resize(segments.size());

		vector<Mat> voting((size_t)kSeg);
		for(auto& vot: voting)
			vot = Mat(segments[0].size(), CV_32FC1, Scalar::all(0));

		for(int v=0; v<segments.size(); ++v){
			const int* pSeg = (int*)segments[v].data;
			for(auto i=0; i<width * height; ++i){
				const int& label = pSeg[i];
				pixelGroup[label][v].push_back(i);
				voting[label].at<float>(i/width, i%width) += 1.0f;
			}
		}

		const float thres = 0.3;
		for(int sid=0; sid<kSeg; ++sid){
			float kFrame = 0.0f;
			for(auto v=0; v<segments.size(); ++v){
				if(!pixelGroup[sid][v].empty())
					kFrame += 1.0f;
			}
			//CHECK_GT(kFrame, 0.0f) << sid;
			if(kFrame < 1.0f)
				continue;
			const float* pVote = (float*) voting[sid].data;
			for(auto i=0; i<width * height; ++i){
				if(pVote[i] / kFrame > thres)
					regionSpan[sid].push_back(i);
			}
		}

		return kSeg;
	}


	void visualizeSegmentGroup(const std::vector<cv::Mat> &images, const std::vector<std::vector<int> > &pixelGroup,
	                           const std::vector<int> &regionSpan){
		CHECK(!images.empty());
		vector<Mat> thumb(images.size());
		for(auto i=0; i<images.size(); ++i)
			cv::pyrDown(images[i], thumb[i]);

		auto dsCoord = [&](const int id){
			const int width = images[0].cols;
			return cv::Point(id%width/2, id/width/2);
		};

		int index = 0;
		Mat regionMask(thumb[0].size(), CV_8UC3, cv::Scalar(255,0,0));
		for(auto pid: regionSpan)
			regionMask.at<Vec3b>(dsCoord(pid)) = cv::Vec3b(0,0,255);



		vector<Mat> segMask(thumb.size());
		for(auto v=0; v<segMask.size(); ++v){
			segMask[v] = Mat(thumb[v].size(), CV_8UC3, cv::Scalar(255,0,0));
			for(auto pid: pixelGroup[v])
				segMask[v].at<Vec3b>(dsCoord(pid)) = Vec3b(0,0,255);
		}

		double blend_weight = 0.4;
		cv::namedWindow("segment");
		while(true){
			Mat segment, region;
			cv::addWeighted(thumb[index], blend_weight, segMask[index], 1.0 - blend_weight, 0.0, segment);
			cv::addWeighted(thumb[index], blend_weight, regionMask, 1.0 - blend_weight, 0.0, region);
			cv::Mat comb;
			cv::hconcat(segment, region, comb);
			cv::imshow("segment", comb);
			if((char)cv::waitKey(33) == 'q')
				break;
			index = (index + 1) % (int)images.size();
		}
		cv::destroyAllWindows();
	}

}//namespace dynamic_stereo