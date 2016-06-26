/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#define _copysign copysign

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include "HoG.h"

// structure for image patch
struct PatchFeature {
	PatchFeature() {}

	cv::Rect roi;
	std::vector<cv::Point> center;

	std::vector<cv::Mat> vPatch;
	void print() const {
		std::cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
		for(unsigned int i=0; i<center.size(); ++i) std::cout << " " << center[i].x << " " << center[i].y; std::cout << std::endl;
	}
	void show(int delay) const;
};

static HoG hog; 

class CRPatch {
public:
	CRPatch(cv::RNG* pRNG, int w, int h, int num_l) : cvRNG(pRNG), width(w), height(h) { vLPatches.resize(num_l);}

	// Extract patches from image
	void extractPatches(const cv::Mat& img, unsigned int n, int label, const cv::Rect* box = 0, const std::vector<cv::Point>* vCenter = 0);

	// Extract features from image
	static void extractFeatureChannels(const cv::Mat& img, std::vector<cv::Mat>& vImg);

	// min/max filter
	static void maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(cv::Mat &src, unsigned int width);
	static void maxfilt(cv::Mat &src, cv::Mat &dst, unsigned int width);
	static void minfilt(cv::Mat &src, unsigned int width);
	static void minfilt(cv::Mat &src, cv::Mat &dst, unsigned int width);

	std::vector<std::vector<PatchFeature> > vLPatches;
private:
	cv::RNG *cvRNG;
	int width;
	int height;
};

