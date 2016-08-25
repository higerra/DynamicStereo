//
// Created by yanhang on 5/24/16.
//

#include "regiondescriptor.h"
#include "../base/utility.h"

#include <fstream>
#include <numeric>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
	namespace ML{
		//feature based over segments
		void extractFeature(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& gradient,
							const cv::Mat& segments, const cv::Mat& mask, TrainSet& trainSet){
			CHECK(!images.empty());
			CHECK(segments.data);
			CHECK_EQ(images[0].size(), segments.size());

			if(trainSet.empty())
				trainSet.resize(2);

			vector<PixelGroup> pixelGroup;
			printf("Regrouping...\n");
			const int kSeg = regroupSegments(segments, pixelGroup);
            for(auto i=0; i<pixelGroup.size(); ++i){
                printf("Group %d, size: %d\n", i, (int)pixelGroup[i].size());
            }
			printf("Assigning label...\n");
			vector<int> segmentLabel;
			if(mask.data) {
				CHECK_EQ(mask.size(), images[0].size());
				assignSegmentLabel(pixelGroup, mask, segmentLabel);
			}else{
				segmentLabel.resize(pixelGroup.size(), 0);
			}

			vector<Mat> colorImage(images.size());
			for(auto i=0; i<images.size(); ++i){
				Mat tmp = images[i] / 255.0;
				cvtColor(tmp, colorImage[i], CV_BGR2Lab);
			}

			//samples are based on segments. Color changes are sample from region
			const int width = images[0].cols;
			const int height = images[0].rows;
			int sid = 0;

			printf("Extracting feature...\n");
			for(const auto& pg: pixelGroup){
                CHECK(!pg.empty());
				SegmentFeature curSample;
				curSample.id = sid;
				curSample.feature.clear();

				//mean and variance in L*a*b
				vector<float> desc_color;
				computeColor(colorImage, pg, desc_color);
				curSample.feature.insert(curSample.feature.end(), desc_color.begin(), desc_color.end());
				//HoG feature
//				vector<float> hog;
//				computeHoG(gradient, pg, hog, kBinHoG);
//				curSample.feature.insert(curSample.feature.end(), hog.begin(), hog.end());
				//shape
				vector<float> desc_shape;
				computeShape(pg, width, height, desc_shape);
				curSample.feature.insert(curSample.feature.end(), desc_shape.begin(), desc_shape.end());

				vector<float> desc_position;
				computePosition(pg, width, height, desc_position);
				curSample.feature.insert(curSample.feature.end(), desc_position.begin(), desc_position.end());

				trainSet[segmentLabel[sid]].push_back(curSample);
				sid++;
			}
		}

		int compressSegments(std::vector<cv::Mat>& segments){
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

			//reassign segment id, update kSeg
			kSeg = 0;
			for(auto& seg: segments){
				int* pSeg = (int*) seg.data;
				for(int i=0; i<seg.cols * seg.rows; ++i){
					pSeg[i] = compressedId[pSeg[i]] - 1;
					kSeg = std::max(kSeg, pSeg[i]);
				}
			}

			return kSeg;
		}

		int regroupSegments(const cv::Mat &segments,
		                    std::vector<PixelGroup> &pixelGroup) {
			CHECK(segments.data);
			CHECK_EQ(segments.type(), CV_32S);

			const int width = segments.cols;
			const int height = segments.rows;

			int kSeg = 0;
			double minid, maxid;
			cv::minMaxLoc(segments, &minid, &maxid);
			kSeg = std::max(kSeg, (int) maxid);
			kSeg++;
			pixelGroup.resize((size_t) kSeg);

			//pixels with the same label are not garranted to be contigueous. Thus for each label,
			//take the contiguous largest contiguous region
			for(auto l=0; l<kSeg; ++l){
				Mat binMask(segments.size(), CV_8UC1, Scalar::all(0));
				for(auto y=0; y<height; ++y){
					for(auto x=0; x<width; ++x){
						if(segments.at<int>(y,x) == l)
							binMask.at<uchar>(y,x) = (uchar)255;
					}
				}
				Mat com, stats, centroid;
				cv::connectedComponentsWithStats(binMask, com, stats, centroid);
				int maxArea = -1, bestLabel = -1;
				for(auto i=1; i<stats.rows; ++i){
					int area = stats.at<int>(i, cv::CC_STAT_AREA);
					if(area > maxArea){
						maxArea = area;
						bestLabel = i;
					}
				}
				for(auto y=stats.at<int>(bestLabel, cv::CC_STAT_TOP);
						y < stats.at<int>(bestLabel, CC_STAT_TOP) + stats.at<int>(bestLabel, CC_STAT_HEIGHT); ++y){
					for(auto x=stats.at<int>(bestLabel, cv::CC_STAT_LEFT);
						x < stats.at<int>(bestLabel, CC_STAT_LEFT) + stats.at<int>(bestLabel, CC_STAT_WIDTH); ++x) {
						if(com.at<int>(y,x) == bestLabel) {
							pixelGroup[l].push_back(y * width + x);
						}
					}
				}
			}

			return kSeg;
		}

		void assignSegmentLabel(const std::vector<PixelGroup>& pixelGroup, const cv::Mat& mask,
		                        std::vector<int>& label) {
			CHECK(!pixelGroup.empty());
			CHECK_EQ(mask.type(), CV_8UC1);
			label.resize(pixelGroup.size(), 0);

			const int kPix = mask.cols * mask.rows;
			const uchar *pMask = mask.data;
			const float posRatio = 0.5f;
			for (auto sid = 0; sid < pixelGroup.size(); ++sid) {
				float total = 0.0f, pos = 0.0f;
				for (const int pid: pixelGroup[sid]) {
					total += 1.0f;
					if (pMask[pid] > (uchar) 200)
						pos += 1.0f;
				}
				CHECK_GT(total, 0.0f) << sid;
				if (pos / total >= posRatio)
					label[sid] = 1;
			}
		}

		void computeColor(const std::vector<cv::Mat>& colorImage, const PixelGroup& pg,
						  std::vector<float>& desc){
			desc.resize(6, 0.0f);
			const int width = colorImage[0].cols;
			vector<vector<double> > labcolor(3);

			for(auto v=0; v<colorImage.size(); ++v){
				for(auto pid: pg){
					if(colorImage[v].type() == CV_32FC3) {
						Vec3f pix = colorImage[v].at<Vec3f>(pid / width, pid % width);
						labcolor[0].push_back((double)pix[0]);
						labcolor[1].push_back((double)pix[1]);
						labcolor[2].push_back((double)pix[2]);
					}else if(colorImage[v].type() == CV_8UC3){
						Vec3b pix = colorImage[v].at<Vec3b>(pid / width, pid % width);
						labcolor[0].push_back((double)pix[0]);
						labcolor[1].push_back((double)pix[1]);
						labcolor[2].push_back((double)pix[2]);
					}
				}
			}

			CHECK(!labcolor[0].empty());
			for(auto i=0; i<3; ++i){
				double mean = std::accumulate(labcolor[i].begin(), labcolor[i].end(), 0.0);
				mean = mean / (double)labcolor[i].size();
				double var = math_util::variance(labcolor[i], mean);
				desc[i] = (float)mean;
				desc[3+i] = (float)var;
			}
		}

		void computeShape(const PixelGroup& pg, const int width, const int height, std::vector<float>& desc){
			//shape descriptor: [area|convexity|rectangleness|number of polygons]
			desc.resize(4, 0.0f);
            //area
            desc[0] = (float)pg.size() / (float)(width * height);
            Mat binMat(height, width, CV_8UC1, Scalar::all(0));

            vector<cv::Point> pts(pg.size());
            for(auto i=0; i<pg.size(); ++i) {
                pts[i].x = pg[i] % width;
                pts[i].y = pg[i] / width;
                binMat.at<uchar>(pts[i]) = (uchar)255;
            }

            //convexity
            vector<cv::Point> chull;
            cv::convexHull(pts, chull);
            double carea = cv::contourArea(chull);
            if(carea == 0)
                desc[1] = 0;
            else
                desc[1] = (float)pg.size() / (float)carea;

            //rectangleness
            cv::RotatedRect minRect = cv::minAreaRect(pts);
            float mrArea = minRect.size.width * minRect.size.height;
            if(mrArea == 0)
                desc[2] = 0;
            else
                desc[2] = (float)pg.size() / mrArea;

            //number of edge is approximated polygon
            const double approxEpsilon = (double)std::min(width, height) / 150.0;
            vector<vector<cv::Point> > oriContour;
            vector<cv::Point> approxContour;
            cv::findContours(binMat, oriContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            CHECK(!oriContour.empty());
            cv::approxPolyDP(oriContour[0], approxContour, approxEpsilon, true);
            desc[3] = (float)approxContour.size();
		}

		void computePosition(const PixelGroup& pg, const int width, const int height, std::vector<float>& desc){
			desc.resize(2, 0.0f);
			for(auto pid: pg){
				desc[0] += (double)(pid % width) / (double)width;
				desc[1] += (double)(pid / width) / (double)height;
			}
			desc[0] /= (float)pg.size();
			desc[1] /= (float)pg.size();
		}

		void computeHoG(const std::vector<cv::Mat>& gradient, const std::vector<std::vector<int> >& pixelIds,
		                std::vector<float>& hog, const int kBin){
			CHECK(!gradient.empty());
			CHECK_EQ(gradient.size(), pixelIds.size());
			const float pi = 3.1415926f;
			const float binsize = pi * 80 / (float) kBin;
			hog.resize((size_t)kBin, 0.0f);

			const int width = gradient[0].cols;
			const int height = gradient[0].rows;
			for(auto v=0; v<gradient.size(); ++v){
				for(auto pid: pixelIds[v]){
					Vec2f g = gradient[v].at<Vec2f>(pid/width, pid%width);
					float val = g[1] / binsize;
					int bin1 = int(val), bin2 = 0;
					float delta = val - bin1 - 0.5f;
					if (delta < 0) {
						bin2 = bin1 < 1 ? kBin - 1 : bin1 - 1;
						delta = -delta;
					} else
						bin2 = bin1 < kBin - 1 ? bin1 + 1 : 0;
					hog[bin1] += (1 - delta) * g[0];
					hog[bin2] += delta * g[0];
				}
			}
            MLUtility::normalizel2(hog);
			const float cut_thres = 0.1;
			for(auto& v: hog){
				if(v < cut_thres)
					v = 0.0;
			}
            MLUtility::normalizel2(hog);
		}

		void computeLine(const PixelGroup& pg, const std::vector<std::vector<LineUtil::KeyLine> >& lineClusters,
						 std::vector<float>& desc){

		}

		void computeTemporalPattern(const std::vector<cv::Mat>& colorImage, const PixelGroup& pg,
									std::vector<float>& desc){

		}

		void visualizeSegmentLabel(const std::vector<cv::Mat>& images, const cv::Mat& segments,
		                           const std::vector<int>& label){
			CHECK(!images.empty());

			const int kPix = images[0].cols * images[0].rows;

			vector<Mat> labelMask(images.size());
			for(auto & m: labelMask)
				m = Mat(images[0].size(), CV_8UC3, Scalar(255,0,0));
			for(auto v=0; v<labelMask.size(); ++v){
				const int* pSeg = (int*) segments.data;
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


	}//namespace Fature
}//namespace dynamic_stereo