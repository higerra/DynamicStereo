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
		void extractFeature(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& segments, const cv::Mat& mask,
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
			const float pi = 3.1415926f;

//		const int tSeg = 230;
//		visualizeSegmentGroup(images, pixelGroup[tSeg], regionSpan[tSeg]);

			printf("Assigning label...\n");
			vector<int> segmentLabel;
			assignSegmentLabel(pixelGroup, mask, segmentLabel);
//		visualizeSegmentLabel(images, segments, segmentLabel);

			//samples are based on segments. Color changes are sample from region
			printf("Extracting feature...\n");
			const vector<int> kBin{6,6,6};
			const int kBinHoG = 9;
			const int diffBin = std::accumulate(kBin.begin(), kBin.end(), 0);
			const int kChannel = 6+ kBinHoG + diffBin + 4+ 1;

			vector<Mat> colorImage(images.size());

			//two channels: magnitude and orientation
			vector<Mat> gradient(images.size());
			for (auto i = 0; i < images.size(); ++i) {
				Mat tmp;
				images[i].convertTo(tmp, CV_32F);
				Mat gx, gy, gray;
				cvtColor(images[i], gray, CV_BGR2GRAY);
				cv::Sobel(gray, gx, CV_32F, 1, 0);
				cv::Sobel(gray, gy, CV_32F, 0, 1);
				gradient[i].create(images[i].size(), CV_32FC2);
				for(auto y=0; y<gx.rows; ++y){
					for(auto x=0; x<gx.cols; ++x){
						float ix = gx.at<float>(y,x);
						float iy = gy.at<float>(y,x);
						Vec2f pix;
						pix[0] = std::sqrt(ix*ix+iy*iy);
						float tx = ix + std::copysign(0.000001f, ix);
						//normalize atan value to [0,80PI]
						pix[1] = (atan(iy / tx) +  pi / 2.0f) * 80;

						gradient[i].at<Vec2f>(y,x) = pix;
					}
				}

				tmp = tmp / 255.0;
				cvtColor(tmp, colorImage[i], CV_BGR2Lab);
			}

			const int width = colorImage[0].cols;
			const int height = colorImage[0].rows;

			int sid = 0;
			const int HoGOffset = 6;
			const int diffHistOffset = HoGOffset + kBinHoG;
			const int shapeOffset = diffHistOffset + diffBin;
			const int lengthOffset = shapeOffset + 4;
			const ColorSpace cspace(ColorSpace::LAB);

			const int unitCount = (int)pixelGroup.size() / 10;

			for(const auto& pg: pixelGroup){
				if(sid % unitCount == (unitCount - 1))
					cout << '.' << flush;

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

				//HoG feature
				vector<float> hog;
				computeHoG(gradient, pg, hog, kBinHoG);
				for(auto i=HoGOffset; i<HoGOffset + kBinHoG; ++i)
					curSample.feature[i] = hog[i-HoGOffset];

				//histogram of color changes. Different channels are computed independently
				const int stride = (int)colorImage.size() / 2;
				vector<float> binUnit(3, 0.0);
				for (auto i = 0; i < cspace.channel; ++i)
					binUnit[i] = 2 * cspace.range[i] / (float) kBin[i];

				vector<vector<float> > diffHist((size_t)cspace.channel);
				for(auto i=0; i<diffHist.size(); ++i){
					diffHist[i].resize((size_t)kBin[i], 0.0f);
				}

				for (auto pid: regionSpan[sid]) {
					for (auto v = 0; v < colorImage.size() - stride; ++v) {
						const int x = pid % width;
						const int y = pid / width;
						Vec3f diff = colorImage[v + stride].at<Vec3f>(y, x) - colorImage[v].at<Vec3f>(y, x);
						for (auto i = 0; i < cspace.channel; ++i) {
							int binId = std::floor((diff[i] + cspace.range[i]) / binUnit[i]);
							CHECK_GE(binId, 0) << x << ' ' << y << ' ' << v << ' ' << diff[i];
							CHECK_LT(binId, kBin[i]) << x << ' ' << y << ' ' << v << ' ' << diff[i];
							diffHist[i][binId] += 1.0;
						}
					}
				}
				const float cut_value = 0.1;
				int diffId = diffHistOffset;
				for(auto& hist: diffHist) {
					normalizel2(hist);
					for (auto &v: hist) {
						if (v < cut_value)
							v = 0.0;
					}
					normalizel2(hist);
					for (auto v: hist)
						curSample.feature[diffId++] = v;
				}


				//shape
				//notice that area is multiplied by 10 to avoid precision issue
				const double area_ratio = 10.0;
				vector<double> area(colorImage.size()), convexity(colorImage.size());

				double length = 0.0;
				for(auto v=0; v<pg.size(); ++v){
					//total area
					if(pg[v].empty())
						continue;

					area[v] = (double)pg[v].size() * area_ratio / (double)(width * height);

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

				curSample.feature[shapeOffset] = (float)mean_area;
				curSample.feature[shapeOffset+1] = (float)mean_convexity;
				curSample.feature[shapeOffset+2] = (float)var_area;
				curSample.feature[shapeOffset+3] = (float)var_convexity;

				//length
				curSample.feature[lengthOffset] = (float)length;

				trainSet[segmentLabel[sid]].push_back(curSample);
				sid++;
			}
			cout << endl;
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