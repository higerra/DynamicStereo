//
// Created by Yan Hang on 7/7/16.
//

#include "randomforest.h"
#include "../base/utility.h"
#include "../common/descriptor.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

	//feature based over segments
	//6 channels for color: mean and variance in L*a*b
	//24 channels for histogram of color changes.
	//2 channels for shape: mean of variance of area, convexity (area / area of convex hall)
	//9 channels for HOG
	//1 channel for length of segment
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

//		const int tSeg = 230;
//		visualizeSegmentGroup(images, pixelGroup[tSeg], regionSpan[tSeg]);

		vector<int> segmentLabel;
		assignSegmentLabel(pixelGroup, mask, segmentLabel);
		visualizeSegmentLabel(images, segments, segmentLabel);

		//samples are based on segments. Color changes are sample from region
		const int kChannel = 6+24+2+9;

		vector<Mat> colorImage(images.size());
		for (auto i = 0; i < images.size(); ++i) {
			Mat tmp;
			images[i].convertTo(tmp, CV_32F);
			tmp = tmp / 255.0;
			cvtColor(tmp, colorImage[i], CV_BGR2Lab);
		}

		const int width = colorImage[0].cols;
		const int height = colorImage[0].rows;

		int sid = 0;
		const int diffHistOffset = 6;
		const int shapeOffset = diffHistOffset + 24;

		Feature::ColorSpace cspace(Feature::ColorSpace::LAB);

		for(const auto& pg: pixelGroup){
			SegmentFeature curSample;
			curSample.id = sid;
			curSample.feature.resize((size_t)kChannel, 0.0f);
			//mean and variance in L*a*b
			vector<vector<double> > labcolor(3);
			for(auto v=0; v<pg.size(); ++v){
				for(auto pid: pg[v]){
					Vec3f pix = colorImage[v].at<Vec3f>(pid/width, pid%width);
					labcolor[0].push_back((double)pix[0]);
					labcolor[1].push_back((double)pix[1]);
					labcolor[2].push_back((double)pix[2]);
				}
			}

			CHECK(!labcolor[0].empty());
			for(auto i=0; i<3; ++i){
				double mean = std::accumulate(labcolor[i].begin(), labcolor[i].end(), 0.0);
				mean = mean / (double)labcolor[i].size();
				double var = math_util::variance(labcolor[i], mean);
				curSample.feature[i] = (float)mean;
				curSample.feature[3+i] = (float)var;
			}

			labcolor.clear();
			//histogram of color changes

			sid++;
		}


	}

	void compressSegments(std::vector<cv::Mat>& segments){
		int kSeg = 0;
		for(const auto& seg: segments){
			double minid, maxid;
			cv::minMaxLoc(seg, &minid, &maxid);
			kSeg = std::max(kSeg, (int)maxid);
		}
		kSeg++;

		//integral vector
		vector<int> compressedId((size_t)kSeg, 0);

		for(const auto& seg: segments){
			for(auto y=0; y<seg.rows; ++y){
				for(auto x=0; x<seg.cols; ++x){
					int l = seg.at<int>(y,x);
					compressedId[l] = 1;
				}
			}
		}

		for(auto i=1; i<compressedId.size(); ++i){
			compressedId[i] += compressedId[i-1];
		}

		for(auto& seg: segments){
			int* pSeg = (int*) seg.data;
			for(int i=0; i<seg.cols * seg.rows; ++i){
				pSeg[i] = compressedId[pSeg[i]] - 1;
			}
		}
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
			CHECK_GT(kFrame, 0.0f) << sid;
			const float* pVote = (float*) voting[sid].data;
			for(auto i=0; i<width * height; ++i){
				if(pVote[i] / kFrame > thres)
					regionSpan[sid].push_back(i);
			}
		}

		return kSeg;
	}

	void assignSegmentLabel(const std::vector<std::vector<std::vector<int> > >& pixelGroup, const cv::Mat& mask,
	                        std::vector<int>& label){
		CHECK(!pixelGroup.empty());
		CHECK_EQ(mask.type(), CV_8UC1);
		label.resize(pixelGroup.size(), 0);

		const int kPix = mask.cols * mask.rows;
		const uchar* pMask = mask.data;
		const float posRatio = 0.5f;
		for(auto sid=0; sid<pixelGroup.size(); ++sid){
			float total = 0.0f, pos = 0.0f;
			for(auto v=0; v<pixelGroup[sid].size(); ++v){
				for(const int pid: pixelGroup[sid][v]){
					total += 1.0f;
					CHECK_LT(pid, kPix);
					if(pMask[pid] > (uchar)200)
						pos += 1.0f;
				}
			}
			CHECK_GT(total, 0.0f) << sid;
			if(pos / total >= posRatio)
				label[sid] = 1;
		}
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

	void visualizeSegmentLabel(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& segments,
	                           const std::vector<int>& label){
		CHECK(!images.empty());

		const int kPix = images[0].cols * images[0].rows;

		vector<Mat> labelMask(images.size());
		for(auto & m: labelMask)
			m = Mat(images[0].size(), CV_8UC3, Scalar(255,0,0));
		for(auto v=0; v<labelMask.size(); ++v){
			const int* pSeg = (int*) segments[v].data;
			uchar* pVis = labelMask[v].data;
			for(auto i=0; i<kPix; ++i){
				if(label[pSeg[i]] == 1){
					pVis[i*3] = (uchar)0;
					pVis[i*3+2] = (uchar)255;
				}
			}
		}

		int index = 0;
		const double blend_weight = 0.4;
		cv::namedWindow("Label");
		while(true){
			Mat vis;
			cv::addWeighted(images[index], blend_weight, labelMask[index], 1.0-blend_weight, 0.0, vis);
			imshow("Label", vis);
			if((uchar)cv::waitKey(33) == 'q')
				break;
			index = (index + 1) % static_cast<int>(images.size());
		}
	}
}//namespace dynamic_stereo