//
// Created by yanhang on 4/29/16.
//

#include <unordered_set>
#include "dynamicsegment.h"
#include "../base/thread_guard.h"
#include "../VisualWord/visualword.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

	void segmentDisplay(const FileIO& file_io, const int anchor,
	                    const std::vector<cv::Mat> &input, const cv::Mat& inputMask,
	                    const string& classifierPath, const string& codebookPath, cv::Mat& result){
		CHECK(!input.empty());
		char buffer[1024] = {};
		const int width = input[0].cols;
		const int height = input[0].rows;

		//display region
		sprintf(buffer, "%s/midres/classification%05d.png", file_io.getDirectory().c_str(), anchor);
		Mat preSeg = imread(buffer, false);

		if(!preSeg.data) {
            const vector<float> levelList{10.0, 20.0, 30.0};
            cv::Ptr<ml::StatModel> classifier;
            Mat codebook;
            VisualWord::VisualWordOption vw_option;
            cv::FileStorage codebookIn(codebookPath, FileStorage::READ);
            CHECK(codebookIn.isOpened()) << "Can not open code book: " << codebookPath;
            codebookIn["codebook"] >> codebook;

            int pixeldesc = (int)codebookIn["pixeldesc"];
            int classifiertype = (int)codebookIn["classifiertype"];
            printf("pixeldesc: %d, classifiertype: %d\n", pixeldesc, classifiertype);
            vw_option.pixDesc = (VisualWord::PixelDescriptor) pixeldesc;
            vw_option.classifierType = (VisualWord::ClassifierType) classifiertype;
            if(classifiertype == VisualWord::RANDOM_FOREST) {
                classifier = ml::RTrees::load<ml::RTrees>(classifierPath);
                cout << "Tree depth: " << classifier.dynamicCast<ml::RTrees>()->getMaxDepth() << endl;
            }
            else if(classifiertype == VisualWord::BOOSTED_TREE)
                classifier = ml::Boost::load<ml::Boost>(classifierPath);
            else if(classifiertype == VisualWord::SVM)
                classifier = ml::SVM::load<ml::SVM>(classifierPath);
            CHECK(classifier.get()) << "Can not open classifier: " << classifierPath;
            VisualWord::detectVideo(input, classifier, codebook, levelList, preSeg, vw_option);
            imwrite(buffer, preSeg);
		};

		const int rh = 5;
		cv::dilate(preSeg,preSeg,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(rh,rh)));
		cv::erode(preSeg,preSeg,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(rh,rh)));

		sprintf(buffer, "%s/temp/segment_display.jpg", file_io.getDirectory().c_str());
		imwrite(buffer, preSeg);

		result = localRefinement(input, preSeg);
	}

	void filterBoudary(const std::vector<cv::Mat>& images, const std::vector<cv::Mat> &seg, cv::Mat &input){
		char buffer[1024] = {};
		CHECK(!seg.empty());
		CHECK_EQ(input.type(), CV_8UC1);

		const int width = seg[0].cols;
		const int height = seg[0].rows;
		const int kPix = width * height;

		int nLabel = -1;
		//get number of labels
		for(auto i=0; i<seg.size(); ++i){
			double minl, maxl;
			cv::minMaxLoc(seg[i], &minl, &maxl);
			nLabel = std::max(nLabel, (int)maxl);
		}
		nLabel++;

		//group pixel to labels for fast query
		vector< vector<float> > posNum(seg.size());
		vector< vector<float> > totalNum(seg.size());
		for(auto i=0; i<seg.size(); ++i){
			posNum[i].resize((size_t)nLabel, 0.0);
			totalNum[i].resize((size_t)nLabel, 0.0);
			for(auto pid=0; pid < width * height; ++pid){
				int segId = seg[i].at<int>(pid/width, pid%width);
				CHECK_LT(segId, nLabel);
//				pixelGroups[segId].push_back(i*kPix+pid);
				totalNum[i][segId] += 1.0f;
				if(input.data[pid] > 200)
					posNum[i][segId] += 1.0f;
			}
		}

		vector<vector<float> > segConf((size_t)kPix);
		for(auto& s: segConf)
			s.resize(seg.size(), 0.0f);

		for(auto v=0; v<seg.size(); ++v){
			for(auto pid=0; pid<width * height; ++pid){
				int label = seg[v].at<int>(pid/width, pid%width);
				CHECK_GT(totalNum[v][label], 0);
				segConf[pid][v] = posNum[v][label] / totalNum[v][label];
			}
		}

		//for each positive pixels compute average number of positive samples in all segments it belongs
		Mat conf(height, width, CV_32F, Scalar::all(0));
		float* pConf = (float *)conf.data;
		const size_t kth = seg.size() / 2;
		for(auto pid=0; pid<height * width; ++pid){
			if(input.data[pid] > 200){
				nth_element(segConf[pid].begin(), segConf[pid].begin()+kth, segConf[pid].end());
				pConf[pid] = segConf[pid][kth];
			}
		}
		const float thres = 0.2;
		for(auto i=0; i<kPix; ++i){
			if(pConf[i] > thres)
				input.data[i] = 255;
			else
				input.data[i] = 0;
		}
	}


	void filterBySegnet(const std::vector<cv::Mat>& images, const std::vector<cv::Mat>& videoSeg, const cv::Mat& segMask, cv::Mat& inputMask){
		CHECK_EQ(segMask.size(), inputMask.size());

		const double ratio_margin = 0.3;
		int nLabel = 0;
		for(auto v=0; v<videoSeg.size(); ++v){
			double minl, maxl;
			cv::minMaxLoc(videoSeg[v], &minl, &maxl);
			nLabel = std::max(nLabel, (int)maxl+1);
		}
		vector<float> invalidCount((size_t)nLabel, 0.0);
		vector<float> segSize((size_t)nLabel, 0.0);

		for(auto v=0; v<videoSeg.size(); ++v){
			for(auto y=0; y<videoSeg[v].rows; ++y){
				for(auto x=0; x<videoSeg[v].cols; ++x){
					if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
						continue;
					int l = videoSeg[v].at<int>(y,x);
					segSize[l] += 1.0;
				}
			}
		}

		for(auto y=0; y<segMask.rows; ++y){
			for(auto x=0; x<segMask.cols; ++x){
				if(segMask.at<uchar>(y,x) < 200){
					for(auto v=0; v<videoSeg.size(); ++v) {
						if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
							continue;
						invalidCount[videoSeg[v].at<int>(y, x)] += 1.0;
					}
				}
			}
		}

		for(auto y=0; y<inputMask.rows; ++y){
			for(auto x=0; x<inputMask.cols; ++x){
				if(inputMask.at<uchar>(y,x) > 200){
					for(auto v=0; v<videoSeg.size(); ++v){
						if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
							continue;
						int sid = videoSeg[v].at<int>(y,x);
						if(segSize[sid] > 0){
							if(invalidCount[sid] / segSize[sid] > ratio_margin){
								inputMask.at<uchar>(y,x) = 0;
								break;
							}
						}
					}
				}
			}
		}


	}

	void groupPixel(const cv::Mat& labels, std::vector<std::vector<Eigen::Vector2d> >& segments){
		CHECK_NOTNULL(labels.data);
		CHECK_EQ(labels.type(), CV_32S);
		double minl, maxl;
		cv::minMaxLoc(labels, &minl, &maxl);
		CHECK_LT(minl, std::numeric_limits<double>::epsilon());
		const int nLabel = (int)maxl;
		segments.clear();
		segments.resize((size_t)nLabel);
		for(auto y=0; y<labels.rows; ++y){
			for(auto x=0; x<labels.cols; ++x){
				int l = labels.at<int>(y,x);
				if(l > 0)
					segments[l-1].push_back(Vector2d(x,y));
			}
		}
	}

	Mat localRefinement(const std::vector<cv::Mat>& images, cv::Mat& mask){
		CHECK(!images.empty());
		const int width = images[0].cols;
		const int height = images[0].rows;

		Mat resultMask(height, width, CV_8UC1, Scalar::all(0));

		Mat labels, stats, centroid;
		int nLabel = cv::connectedComponentsWithStats(mask, labels, stats, centroid);
		const int* pLabel = (int*) labels.data;

		const int min_area = 50;
		const double maxRatioOcclu = 0.3;

		int kOutputLabel = 1;

		const int testL = -1;

		const int localMargin = std::min(width, height) / 10;
		for(auto l=1; l<nLabel; ++l){
			if(testL > 0 && l != testL)
				continue;

			const int area = stats.at<int>(l, CC_STAT_AREA);
			//search for bounding box.
			const int cx = stats.at<int>(l,CC_STAT_LEFT) + stats.at<int>(l,CC_STAT_WIDTH) / 2;
			const int cy = stats.at<int>(l,CC_STAT_TOP) + stats.at<int>(l,CC_STAT_HEIGHT) / 2;

			printf("========================\n");
			printf("label:%d/%d, centroid:(%d,%d), area:%d\n", l, nLabel, cx, cy, area);
			if(area < min_area) {
				printf("Area too small\n");
				continue;
			}

			//filter out mostly occlued areas
			int nOcclu = 0;
			for(auto y=0; y<height; ++y){
				for(auto x=0; x<width; ++x){
					if(pLabel[y*width+x] != l)
						continue;
					int pixOcclu = 0;
					for(auto v=0; v<images.size(); ++v){
						if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
							pixOcclu++;
					}
					if(pixOcclu > (int)images.size() / 3)
						nOcclu++;
				}
			}
			if(nOcclu > maxRatioOcclu * area) {
				printf("Violate occlusion constraint\n");
				continue;
			}

			const int left = std::max(stats.at<int>(l, CC_STAT_LEFT)-localMargin, 0);
			const int top = std::max(stats.at<int>(l, CC_STAT_TOP)-localMargin, 0);
			int roiw = stats.at<int>(l, CC_STAT_WIDTH) + 2*localMargin;
			int roih = stats.at<int>(l, CC_STAT_HEIGHT) + 2*localMargin;
			if(roiw + left >= width)
				roiw = width - left;
			if(roih + top >= height)
				roih = height - top;

			Mat localGBMask(roih, roiw, CV_8UC1, Scalar::all(GC_PR_BGD));
			Mat bMaskBG(roih, roiw, CV_8UC1, Scalar::all(255));
			for(auto y=0; y<roih; ++y){
				for(auto x=0; x<roiw; ++x){
					if(labels.at<int>(y+top, x+left) == l) {
						localGBMask.at<uchar>(y, x) = GC_FGD;
						bMaskBG.at<uchar>(y,x) = 0;
					}
				}
			}
			cv::erode(bMaskBG, bMaskBG, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(5,5)));
			for(auto y=0; y<roih; ++y){
				for(auto x=0; x<roiw; ++x){
					if(bMaskBG.at<uchar>(y,x) > 200) {
						localGBMask.at<uchar>(y, x) = GC_BGD;
					}
				}
			}

			vector<Mat> localPatches(images.size());
			for(auto v=0; v<images.size(); ++v)
				localPatches[v] = images[v](cv::Rect(left, top, roiw, roih));

			printf("running grabcut...\n");
			mfGrabCut(localPatches, localGBMask);
			printf("done\n");

//			Mat resultVis = localPatches[localPatches.size()/2].clone();
			for (auto y = top; y < top + roih; ++y) {
				for (auto x = left; x < left + roiw; ++x) {
					if (localGBMask.at<uchar>(y - top, x - left) == GC_PR_FGD ||
						localGBMask.at<uchar>(y - top, x - left) == GC_FGD) {
						resultMask.at<uchar>(y, x) = 255;
//						resultVis.at<Vec3b>(y-top, x-left) = resultVis.at<Vec3b>(y-top, x-left) / 2 + Vec3b(0, 0, 128);
					} else {
//						resultVis.at<Vec3b>(y-top, x-left) = resultVis.at<Vec3b>(y-top, x-left) / 2 + Vec3b(128, 0, 0);
					}
				}
			}
//			imshow("Result of grabcut", resultVis);
//			waitKey(0);
			kOutputLabel++;
		}

		Mat result;
		cv::connectedComponents(resultMask, result);
		return result;
	}

}//namespace dynamic_stereo
