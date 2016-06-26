/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"


class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(const cv::Mat& img, std::vector<std::vector<cv::Mat> >& imgDetect, std::vector<float>& ratios);

	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

private:
	void detectColor(const cv::Mat& img, std::vector<cv::Mat>& imgDetect, std::vector<float>& ratios);

	const CRForest* crForest;
	int width;
	int height;
};
