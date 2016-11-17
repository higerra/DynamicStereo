//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"
#include "../base/utility.h"
#include "../GeometryModule/dynamic_utility.h"
#include "external/MRF2.2/GCoptimization.h"
#include "../VideoSegmentation/videosegmentation.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

	void computeFrequencyConfidence(const std::vector<cv::Mat> &warppedImg, const float threshold, Depth &result, cv::Mat& frq_ranges){
		CHECK(!warppedImg.empty());
		const int width = warppedImg[0].cols;
		const int height = warppedImg[0].rows;
		result.initialize(width, height, 0.0);
        frq_ranges.create(height, width, CV_32SC2);
        frq_ranges.setTo(cv::Scalar::all(0));

		const double alpha = 2, beta = 2.5;
		const int min_frq = 6;
		const int tx = -1, ty= -1;

        const int N = (int)warppedImg.size();

		vector<Mat> smoothed(warppedImg.size());
		for(auto v=0; v<warppedImg.size(); ++v){
			cv::blur(warppedImg[v], smoothed[v], cv::Size(3,3));
		}

		//build intensity map. Pixel with low intensities should not be considered flashing
		Mat intensity_map(height, width, CV_8UC1, Scalar::all(0));
//		{
//			const size_t kth = 0.9 * N;
//			vector<vector<uchar> > intensity_array(width * height);
//			for(auto& ia: intensity_array){
//				ia.resize(N, (uchar)0);
//			}
//
//			for(auto v=0; v<N; ++v){
//				Mat gray;
//				cvtColor(smoothed[v], gray, CV_BGR2GRAY);
//				for(auto i=0; i < width * height; ++i){
//					intensity_array[i][v] = gray.data[i];
//				}
//			}
//
//			for(auto i=0; i<width * height; ++i){
//				std::nth_element(intensity_array[i].begin(), intensity_array[i].begin() + kth, intensity_array[i].end());
//				intensity_map.data[i] = intensity_array[i][kth];
//			}
//		}

        //Divide the visible range to kInt intervals, exhaustively search for all ranges
        const int kInt = 10;
        const int minInt = kInt / 2;
        const int interval = N / kInt;
		const uchar intensity_threshold = static_cast<uchar>(255.0*0.5);

        auto sigmoid = [&](const double x){
            return 1 / (1 + std::exp(-1*alpha*(x - beta)));
        };
#pragma omp parallel for
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
//				if(intensity_map.at<uchar>(y,x) < intensity_threshold){
//					result(x,y) = 0;
//					continue;
//				}
				Mat colorArray = Mat(3, N, CV_32FC1, Scalar::all(0));
				float* pArray = (float*)colorArray.data;
				for(auto v=0; v<N; ++v) {
					Vec3b pixv = smoothed[v].at<Vec3b>(y, x);
					pArray[v] = (float) pixv[0];
					pArray[N + v] = (float) pixv[1];
					pArray[2 * N + v] = (float) pixv[2];
				}

                Mat sumMat = Mat(3, N+1, CV_32FC1, Scalar::all(0));
                for(auto i=1; i<sumMat.cols; ++i)
                    sumMat.col(i) = sumMat.col(i-1) + colorArray.col(i-1);

                //start searching
                double frqConf = -1;
                for(auto sid =0; sid < minInt * interval; sid += interval){
                    for(auto eid = sid + minInt * interval; eid < N; eid += interval){
                        Mat curArray;
                        colorArray.colRange(sid, eid+1).copyTo(curArray);
                        Mat meanColor = (sumMat.col(eid+1) - sumMat.col(sid)) / (float)(eid-sid+1);
                        for(auto i=0; i<curArray.cols; ++i)
                            curArray.col(i) -= meanColor;
                        double curConf = sigmoid(utility::getFrequencyScore(curArray, min_frq));
                        if(curConf > frqConf){
                            frqConf = curConf;
                        }
                        if(curConf > threshold){
                            frq_ranges.at<Vec2i>(y,x)[0] = std::min(frq_ranges.at<Vec2i>(y,x)[0], sid);
                            frq_ranges.at<Vec2i>(y,x)[1] = std::max(frq_ranges.at<Vec2i>(y,x)[1], eid);
                        }
                    }
                }
                result(x,y) = frqConf;
			}
		}

	}


	void segmentFlashy(const FileIO& file_io, const int anchor,
	                   const std::vector<cv::Mat> &input,
                       std::vector<std::vector<Eigen::Vector2i> >& segments_flashy,
                       std::vector<Eigen::Vector2i>& ranges) {
        char buffer[128] = {};
        const int width = input[0].cols;
        const int height = input[0].rows;

        Mat preSeg(height, width, CV_8UC1, Scalar::all(0));
        //repetative pattern
        printf("Computing frequency confidence...\n");
        double freThreshold = 0.7;
        Depth frequency;
        Mat frq_range;
        computeFrequencyConfidence(input, freThreshold, frequency, frq_range);
        printf("Done\n");
        sprintf(buffer, "%s/temp/conf_frquency_average%05d.jpg", file_io.getDirectory().c_str(), anchor);
        frequency.saveImage(string(buffer), 255);

        for (auto i = 0; i < width * height; ++i) {
            if (frequency[i] > freThreshold)
                preSeg.data[i] = (uchar) 255;
        }

        const int rh = 6;
        const int rl = 2;
        Mat processed;
        cv::erode(preSeg, processed, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rl, rl)));
        cv::dilate(processed, processed, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rh, rh)));
        sprintf(buffer, "%s/mid/segment_flashy%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, processed);

        Mat labels;
        cv::connectedComponents(processed, labels);
        groupPixel(labels, segments_flashy);
        ranges.resize(segments_flashy.size(), Vector2i(-1,input.size()));

        for(auto sid=0; sid<segments_flashy.size(); ++sid){
            for(const auto& pid: segments_flashy[sid]){
                if(preSeg.at<uchar>(pid[1], pid[0]) > (uchar)200){
                    ranges[sid][0] = std::max(ranges[sid][0], frq_range.at<Vec2i>(pid[1], pid[0])[0]);
                    ranges[sid][1] = std::min(ranges[sid][1], frq_range.at<Vec2i>(pid[1], pid[0])[1]);
                }
            }
        }
    }



//	void DynamicSegment::solveMRF(const std::vector<double> &unary,
//								  const std::vector<double>& vCue,
//								  const std::vector<double>& hCue,
//								  const cv::Mat& img, const double weight_smooth,
//								  cv::Mat& result) const {
//		const int width = img.cols;
//		const int height = img.rows;
//
//		double MRF_smooth[] = {0,weight_smooth,weight_smooth,0};
//
//		DataCost *dataCost = new DataCost(const_cast<double*>(unary.data()));
//		SmoothnessCost *smoothnessCost = new SmoothnessCost(MRF_smooth, const_cast<double*>(hCue.data()), const_cast<double*>(vCue.data()));
//		EnergyFunction* energy_function = new EnergyFunction(dataCost, smoothnessCost);
//
//		Expansion mrf(width,height,2,energy_function);
//		mrf.initialize();
//		mrf.clearAnswer();
//		for(auto i=0; i<width*height; ++i)
//			mrf.setLabel(i,0);
//		double initDataE = mrf.dataEnergy();
//		double initSmoothE = mrf.smoothnessEnergy();
//		float mrf_time;
//		mrf.optimize(10, mrf_time);
//		printf("Inital energy: (%.3f,%.3f,%.3f), final energy: (%.3f,%.3f,%.3f), time:%.2fs\n", initDataE, initSmoothE, initDataE+initSmoothE,
//		       mrf.dataEnergy(), mrf.smoothnessEnergy(), mrf.dataEnergy()+mrf.smoothnessEnergy(), mrf_time);
//
//		result = Mat(height, width, CV_8UC1);
//		uchar* pResult = result.data;
//		for(auto i=0; i<width*height; ++i){
//			if(mrf.getLabel(i) > 0)
//				pResult[i] = (uchar)255;
//			else
//				pResult[i] = (uchar)0;
//		}
//		delete dataCost;
//		delete smoothnessCost;
//		delete energy_function;
//
//	}

}//namespace dynamic_stereo
