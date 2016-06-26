/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class HoG {
public:
	HoG();
	~HoG() {delete ptGauss;}
	void extractOBin(const cv::Mat &Iorient, const cv::Mat& Imagn, std::vector<cv::Mat>& out, int off);
private:

	void calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, std::vector<double>& desc);
	void binning(float v, float w, std::vector<double>&, int maxb);

	int bins;
	float binsize; 

	int g_w;
	cv::Mat Gauss;

	// Gauss as vector
	float* ptGauss;
};

inline void HoG::calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, std::vector<double>& desc) {
	for(int i=0; i<bins;i++)
		desc[i]=0;

	uchar* ptO = &ptOrient[0];
	uchar* ptM = &ptMagn[0];
	int i=0;
	for(int y=0;y<g_w; ++y, ptO+=step, ptM+=step) {
		for(int x=0;x<g_w; ++x, ++i) {
			binning((float)ptO[x]/binsize, (float)ptM[x] * ptGauss[i], desc, bins);
		}
	}
}

inline void HoG::binning(float v, float w, std::vector<double>& desc, int maxb) {
	int bin1 = int(v);
	int bin2;
	float delta = v-bin1-0.5f;
	if(delta<0) {
		bin2 = bin1 < 1 ? maxb-1 : bin1-1; 
		delta = -delta;
	} else
		bin2 = bin1 < maxb-1 ? bin1+1 : 0; 
	desc[bin1] += (1-delta)*w;
	desc[bin2] += delta*w;
}

