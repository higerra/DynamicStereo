/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>

using namespace std;
using namespace cv;

void CRForestDetector::detectColor(const cv::Mat& img, vector<Mat>& imgDetect, std::vector<float>& ratios) {

	// extract features
	vector<Mat> vImg;
	CRPatch::extractFeatureChannels(img, vImg);

	// reset output image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		imgDetect[c].setTo(Scalar::all(0));
		//cvSetZero( imgDetect[c] );

	// get pointers to feature channels
	int stepImg;
	vector<uchar*> ptFCh(vImg.size());
	vector<uchar*> ptFCh_row(vImg.size());
	for(unsigned int c=0; c<vImg.size(); ++c) {
		ptFCh[c] = vImg[c].data;
//		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
//	stepImg /= sizeof(ptFCh[0][0]);
	stepImg = vImg[0].cols;

	// get pointer to output image
	int stepDet;
	vector<float*> ptDet(imgDetect.size());
	for(unsigned int c=0; c<imgDetect.size(); ++c)
		ptDet[c] = (float*) imgDetect[c].data;
//		cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
//	stepDet /= sizeof(ptDet[0][0]);
	stepDet = imgDetect[0].cols;

	int xoffset = width/2;
	int yoffset = height/2;
	
	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset; 

	for(y=0; y<img.rows-height; ++y, ++cy) {
		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x<img.cols-width; ++x, ++cx) {

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row.data(), stepImg);
			
			// vote for all trees (leafs) 
			for(vector<const LeafNode*>::const_iterator itL = result.begin(); itL!=result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
			        // with a probability for foreground > 0.5
			        // 
				// if((*itL)->pfg>0.5) {

					// voting weight for leaf 
					float w = (*itL)->pfg / float( (*itL)->vCenter.size() * result.size() );

					// vote for all points stored in the leaf
					for(auto it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it) {
						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  int xx = int(cx - (*it)[0].x * ratios[c] + 0.5);
						  int yy = cy-(*it)[0].y;
						  if(yy>=0 && yy<imgDetect[c].rows && xx>=0 && xx<imgDetect[c].cols) {
						    *(ptDet[c]+xx+yy*stepDet) += w;
						  }
						}
					}

				 // } // end if

			}

			// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// smooth result image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cv::blur(imgDetect[c], imgDetect[c], cv::Size(3,3));
		//cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);
}

void CRForestDetector::detectPyramid(const cv::Mat& img, vector<vector<Mat> >& vImgDetect, std::vector<float>& ratios) {

	if(img.channels() == 1) {
		std::cerr << "Gray color images are not supported." << std::endl;
	} else { // color
		cout << "Timer" << endl;
		float tstart = cv::getTickCount();

		for(int i=0; i<int(vImgDetect.size()); ++i) {
//			IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);
			Mat cLevel(vImgDetect[i][0].size(), CV_8UC3);
			cv::resize(img, cLevel, cLevel.size(), CV_INTER_LINEAR);
			//cvResize( img, cLevel, CV_INTER_LINEAR );

			// detection
			detectColor(cLevel,vImgDetect[i],ratios);
		}

		cout << "Time: " << ((float)getTickCount() - tstart) / (float)getTickFrequency() << " sec" << endl;

	}

}








