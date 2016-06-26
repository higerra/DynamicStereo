/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <vector>
#include <iostream>
#include "HoG.h"

using namespace std;
using namespace cv;

HoG::HoG() {
	bins = 9;
	binsize = (3.14159265f*80.0f)/float(bins);;

	g_w = 5;
	Gauss.create( g_w, g_w, CV_32FC1 );
	double a = -(g_w-1)/2.0;
	double sigma2 = 2*(0.5*g_w)*(0.5*g_w);
	double count = 0;
	for(int x = 0; x<g_w; ++x) {
		for(int y = 0; y<g_w; ++y) {
			double tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
			count += tmp;
			Gauss.at<float>(y,x) = (float)tmp;
			//cvSet2D( Gauss, x, y, cvScalar(tmp) );
		}
	}
	Gauss = Gauss * (1.0/count);
	ptGauss = new float[g_w*g_w];
	int i = 0;
	for(int y = 0; y<g_w; ++y) 
		for(int x = 0; x<g_w; ++x)
			ptGauss[i++] = Gauss.at<float>(y,x);
			//ptGauss[i++] = (float)cvmGet( Gauss, x, y );

}


void HoG::extractOBin(const cv::Mat& Iorient, const cv::Mat& Imagn, std::vector<cv::Mat>& out, int off) {
	vector<double> desc((size_t) bins);

	// reset output image (border=0) and get pointers
	vector<uchar*> ptOut((size_t) bins);
	vector<uchar*> ptOut_row((size_t)bins);

	for(int k=off; k<bins+off; ++k) {
		out[k].setTo(Scalar(0));
		ptOut[k-off] = out[k].data;
		//cvSetZero( out[k] );
		//cvGetRawData( out[k], (uchar**)&(ptOut[k-off]));
	}

	// get pointers to orientation, magnitude
	int step = Iorient.cols;
	uchar* ptOrient = Iorient.data;
	uchar* ptOrient_row;
//	cvGetRawData( Iorient, (uchar**)&(ptOrient), &step);
//	step /= sizeof(ptOrient[0]);

	uchar* ptMagn = Imagn.data;
	uchar* ptMagn_row;
//	cvGetRawData( Imagn, (uchar**)&(ptMagn));

	int off_w = int(g_w/2.0); 
	for(int l=0; l<bins; ++l)
		ptOut[l] += off_w*step;

	for(int y=0;y<Iorient.rows-g_w; y++, ptMagn+=step, ptOrient+=step) {
		// Get row pointers
		ptOrient_row = &ptOrient[0];
		ptMagn_row = &ptMagn[0];
		for(int l=0; l<bins; ++l)
			ptOut_row[l] = &ptOut[l][0]+off_w;

		for(int x=0; x<Iorient.cols-g_w; ++x, ++ptOrient_row, ++ptMagn_row) {
		
			calcHoGBin( ptOrient_row, ptMagn_row, step, desc );

			for(int l=0; l<bins; ++l) {
				*ptOut_row[l] = (uchar)desc[l];
				++ptOut_row[l];
			}
		}

		// update pointer
		for(int l=0; l<bins; ++l)
			ptOut[l] += step;
	}
}



