//
// Created by yanhang on 5/24/16.
//

#include "regiondescriptor.h"
#include "../base/utility.h"

#include <numeric>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
	namespace Feature{

		//feature based over segments
		//6 channels for color: mean and variance in L*a*b
		//9 channels for HoG in RGB
		//18 channels for histogram of color changes in L*a*b.
		//4 channels for shape: mean of variance of area, convexity (area / area of convex hall)
		//1 channel for length of segment
		//4 channels for the centroid position (mean, variance)
		void extractFeature(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& gradient,
							const std::vector<cv::Mat>& segments, const cv::Mat& mask,
		                    const FeatureOption& option, TrainSet& trainSet){
			CHECK(!images.empty());
			CHECK(!segments.empty());
			CHECK_EQ(images.size(), segments.size());
			CHECK_EQ(images[0].size(), segments[0].size());

			if(trainSet.empty())
				trainSet.resize(2);

			vector<vector<vector<int> > > pixelGroup;
			vector<vector<int> > regionSpan;
			printf("Regrouping...\n");
			const int kSeg = regroupSegments(segments, pixelGroup, regionSpan);

//			for(auto tSeg=0; tSeg<kSeg; ++tSeg)
//				visualizeSegmentGroup(images, pixelGroup[tSeg], regionSpan[tSeg]);

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

//		visualizeSegmentLabel(images, segments, segmentLabel);

			//samples are based on segments. Color changes are sample from region

			const vector<int> kBin{8,8,8};
			const int kBinHoG = 9;
			const int diffBin = std::accumulate(kBin.begin(), kBin.end(), 0);
			const int kChannel = 6+ kBinHoG + diffBin + 4+ 1+ 4;

			const int width = images[0].cols;
			const int height = images[0].rows;
			int sid = 0;

			const int unitCount = (int)pixelGroup.size() / 10;

			printf("Extracting feature...\n");
			for(const auto& pg: pixelGroup){
				if(sid % unitCount == (unitCount - 1))
					cout << '.' << flush;
				SegmentFeature curSample;
				curSample.id = sid;
				curSample.feature.clear();

				//mean and variance in L*a*b
				vector<float> desc_color;
				computeColor(colorImage, pg, desc_color);
				curSample.feature.insert(curSample.feature.end(), desc_color.begin(), desc_color.end());

				//HoG feature
				vector<float> hog;
				computeHoG(gradient, pg, hog, kBinHoG);
				curSample.feature.insert(curSample.feature.end(), hog.begin(), hog.end());

				//histogram of color changes. Different channels are computed independently
				vector<float> desc_change;
				computeColorChange(colorImage, regionSpan[sid], kBin, desc_change);
				curSample.feature.insert(curSample.feature.end(), desc_change.begin(), desc_change.end());

				//shape
				vector<float> desc_shape;
				computeShapeAndLength(pg, width, height, desc_shape);
				curSample.feature.insert(curSample.feature.end(), desc_shape.begin(), desc_shape.end());

				//position
				vector<float> desc_position;
				computePosition(pg, width, height, desc_position);
				curSample.feature.insert(curSample.feature.end(), desc_position.begin(), desc_position.end());

				trainSet[segmentLabel[sid]].push_back(curSample);
				sid++;
			}
			cout << endl;
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

			//compute temporal range of a segment, for acceleration
			vector<pair<int, int> > range((size_t)kSeg, std::pair<int,int>((int)segments.size(), -1));
			for(auto v=0; v<segments.size(); ++v){
				const int *pSeg = (int *) segments[v].data;
				for(auto i=0; i<width * height; ++i){
					const int& sid = pSeg[i];
					range[sid].first = std::min(range[sid].first, v);
					range[sid].second = std::max(range[sid].second, v);
				}
			}

			for(auto v=0; v<segments.size(); ++v){
				const int* pSeg = (int*) segments[v].data;
				for(auto i=0; i<width * height; ++i){
					pixelGroup[pSeg[i]][v].push_back(i);
				}
			}

			const float thres = 0.3;
			Mat vote(height, width, CV_32FC1, Scalar::all(0.0f));
			float* pVote = (float*)vote.data;
			for(auto sid=0; sid<kSeg; ++sid) {
				const float kFrame = static_cast<float>(range[sid].second - range[sid].first + 1);
				vote.setTo(Scalar::all(0.0));
				for (int v = range[sid].first; v <= range[sid].second; ++v) {
					for (auto i: pixelGroup[sid][v]){
						pVote[i] += 1.0f;
					}
				}
				CHECK_GT(kFrame, 0.0f) << sid;
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

		void computeColor(const std::vector<cv::Mat>& colorImage, const std::vector<std::vector<int> >& pg,
						  std::vector<float>& desc){
			desc.resize(6, 0.0f);
			const int width = colorImage[0].cols;
			vector<vector<double> > labcolor(3);
			for(auto v=0; v<pg.size(); ++v){
				for(auto pid: pg[v]){
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

		void computeColorChange(const std::vector<cv::Mat>& colorImage, const std::vector<int>& region,
								const std::vector<int>& kBin, std::vector<float>& desc){
            //histogram of color changes. Different channels are computed independently
			const ColorSpace cspace(ColorSpace::LAB);
			CHECK_LE(kBin.size(), cspace.channel);

			const int totalBin = std::accumulate(desc.begin(), desc.end(), 0);
			desc.resize((size_t)totalBin, 0.0f);

			const int width = colorImage[0].cols;
			const int stride = (int)colorImage.size() / 2;
			vector<float> binUnit(kBin.size(), 0.0);
			for (auto i = 0; i < kBin.size(); ++i)
				binUnit[i] = 2 * cspace.range[i] / (float) kBin[i];

			vector<vector<float> > diffHist(kBin.size());
			for(auto i=0; i<diffHist.size(); ++i){
				diffHist[i].resize((size_t)kBin[i], 0.0f);
			}

			for (auto pid: region) {
				for (auto v = 0; v < colorImage.size() - stride; ++v) {
					const int x = pid % width;
					const int y = pid / width;
					Vec3f diff = colorImage[v + stride].at<Vec3f>(y, x) - colorImage[v].at<Vec3f>(y, x);
					for (auto i = 0; i < kBin.size(); ++i) {
						int binId = std::floor((diff[i] + cspace.range[i]) / binUnit[i]);
						CHECK_GE(binId, 0) << x << ' ' << y << ' ' << v << ' ' << diff[i];
						CHECK_LT(binId, kBin[i]) << x << ' ' << y << ' ' << v << ' ' << diff[i];
						diffHist[i][binId] += 1.0;
					}
				}
			}
			const float cut_value = 0.1;
			for(auto& hist: diffHist) {
				normalizel2(hist);
				for (auto &v: hist) {
					if (v < cut_value)
						v = 0.0;
				}
				normalizel2(hist);
				desc.insert(desc.end(), hist.begin(), hist.end());
			}
		}

		void computeShapeAndLength(const std::vector<std::vector<int> >& pg, const int width, const int height, std::vector<float>& desc){
			desc.resize(5, 0.0f);
			double length = 0.0;
			const double area_ratio = 10.0;
			vector<double> area(pg.size()), convexity(pg.size());
			const int kPix = width * height;
			for(auto v=0; v<pg.size(); ++v){
				//total area
				if(pg[v].empty())
					continue;
				area[v] = (double)pg[v].size() * area_ratio / (double)kPix;

				//convexity
				vector<cv::Point> locs(pg[v].size());
				for(auto i=0; i<pg[v].size(); ++i) {
					locs[i].x = pg[v][i] % width;
					locs[i].y = pg[v][i] / width;
				}
				vector<cv::Point> chull;
				cv::convexHull(locs, chull);
				double area_hull = cv::contourArea(chull);
				if(area_hull >= 1)
					convexity[v] = (double)pg[v].size() / area_hull;
				else
					convexity[v] = 0.0;
				length += 1.0;
			}
			CHECK_GT(length, 0.0);
			double mean_area = std::accumulate(area.begin(), area.end(), 0.0) / length;
			double mean_convexity = std::accumulate(convexity.begin(), convexity.end(), 0.0) / length;
			double var_area = 0.0, var_convexity = 0.0;
			if(length > 2.0) {
				var_area = math_util::variance(area, mean_area);
				var_convexity = math_util::variance(convexity, mean_convexity);
			}

			desc[0] = (float)mean_area;
			desc[1] = (float)mean_convexity;
			desc[2] = (float)var_area;
			desc[3] = (float)var_convexity;

			//length
			desc[4] = (float)length;
		}

		void computePosition(const std::vector<std::vector<int> >& pg, const int width, const int height, std::vector<float>& desc){
			desc.resize(4, 0.0f);
			vector<vector<double> > centroid(2);
			for(auto v=0; v<pg.size(); ++v){
				if(pg[v].empty())
					continue;
				double avex = 0.0, avey = 0.0;
				for(auto pid: pg[v]){
					avex += (double)(pid % width) / (double)width;
					avey += (double)(pid / width) / (double)height;
				}
				centroid[0].push_back(avex / (double)pg[v].size());
				centroid[1].push_back(avey / (double)pg[v].size());
			}
			CHECK_GT(centroid[0].size(), 0);
			desc[0] = (float)std::accumulate(centroid[0].begin(), centroid[0].end(), 0.0) / (float)centroid[0].size();
			desc[1] = (float)std::accumulate(centroid[1].begin(), centroid[1].end(), 0.0) / (float)centroid[1].size();
			if(centroid[0].size() == 0){
				desc[2] = 0.0f;
				desc[3] = 0.0f;
			}else {
				desc[2] = (float) math_util::variance(centroid[0], desc[0]);
				desc[3] = (float) math_util::variance(centroid[1], desc[1]);
			}
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
			normalizel2(hog);
			const float cut_thres = 0.1;
			for(auto& v: hog){
				if(v < cut_thres)
					v = 0.0;
			}
			normalizel2(hog);
		}

		void computeGradient(const cv::Mat& image, cv::Mat& gradient){
			const float pi = 3.1415926f;
			Mat gx, gy, gray;
			cvtColor(image, gray, CV_BGR2GRAY);
			cv::Sobel(gray, gx, CV_32F, 1, 0);
			cv::Sobel(gray, gy, CV_32F, 0, 1);
			gradient.create(image.size(), CV_32FC2);
			for(auto y=0; y<gx.rows; ++y){
				for(auto x=0; x<gx.cols; ++x){
					float ix = gx.at<float>(y,x);
					float iy = gy.at<float>(y,x);
					Vec2f pix;
					pix[0] = std::sqrt(ix*ix+iy*iy);
					float tx = ix + std::copysign(0.000001f, ix);
					//normalize atan value to [0,80PI]
					pix[1] = (atan(iy / tx) +  pi / 2.0f) * 80;
					gradient.at<Vec2f>(y,x) = pix;
				}
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
	}//namespace Fature
}//namespace dynamic_stereo