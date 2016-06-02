//
// Created by yanhang on 6/2/16.
//

#include "dynamicsegment.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void mfGrabCut(const std::vector<cv::Mat>& images, cv::Mat& mask){
        //test for grabuct segmentation
        //initial mask
        Mat midMask = mask.clone();

        const int rh = 5;
        cv::dilate(midMask, midMask, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rh, rh)));
        cv::erode(midMask, midMask, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rh, rh)));

//			sprintf(buffer, "%s/midres/gmm_negative%05d.gmm", file_io.getDirectory().c_str(), anchor);
//			cv::Ptr<cv::ml::EM> gmm_negative = cv::ml::EM::create();
//
//
//			//collect negative sample
//			const int bgstride = 4;
//			vector<Vector3d> nsamples;
//			for(auto y=0; y<height; y+=bgstride) {
//				for (auto x = 0; x < width; x += bgstride) {
//					if (pBgmask[y * width + x] > 200) {
//						for (auto v = 0; v < input.size(); v += bgstride) {
//							Vec3b pix = input[v].at<Vec3b>(y, x);
//							nsamples.push_back(Vector3d((double) pix[0], (double) pix[1], (double) pix[2]));
//						}
//					}
//				}
//			}
//			Mat ntrainSample((int)nsamples.size(), 3, CV_64F);
//			for(auto i=0; i<nsamples.size(); ++i){
//				ntrainSample.at<double>(i,0) = nsamples[i][0];
//				ntrainSample.at<double>(i,1) = nsamples[i][1];
//				ntrainSample.at<double>(i,2) = nsamples[i][2];
//			}
//			float bgGMMt = (float)cv::getTickCount();
//			cout << "Estimating background color model, number of samples:" << ntrainSample.rows << endl;
//			gmm_negative->trainEM(ntrainSample);
//			printf("Done. Time usage: %.3f\n", ((float)getTickCount() - bgGMMt) / (float)getTickFrequency());

        //connected component analysis
        Mat labels, stats, centroids;
        int nCom = cv::connectedComponentsWithStats(midMask, labels, stats, centroids);
        const int *pLabel = (int *) labels.data;
        const int localMargin = std::min(width, height) / 20;

        printf("%d connected component\n", nCom);
        //Note: label = 0 represents background
        for (auto l = 1; l < nCom; ++l) {
            //Drop components with area < min_area.
            //For each remaining component, perform grabcut seperately
            printf("Component %d ", l);
            if (stats.at<int>(l, CC_STAT_AREA) < min_area) {
                printf("Area too small(%d), drop\n", stats.at<int>(l, CC_STAT_AREA));
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

            Mat localROI;
            midMask(cv::Rect(left, top, roiw, roih)).copyTo(localROI);
            Mat fgMask, bgMask;

//

            for(auto y=0; y<roih; ++y){
                for(auto x=0; x<roiw; ++x){
                    int oriId = (y+top) * width + x + left;
                    if(pGcmask[y*roiw+x] == GC_FGD || pGcmask[y*roiw+x] == GC_PR_FGD)
                        pResult[oriId] = (uchar)255;
                }
            }


            //estimate GMM
//				cv::Ptr<cv::ml::EM> gmm_positive = cv::ml::EM::create();
//
//				vector<Vector3d> psamples;
//				//collect classifier sample
//				for (auto y = top; y < top + roih; ++y) {
//					for (auto x = left; x < left + roiw; ++x) {
//						if (pLabel[(y + top) * width + x + left] == l) {
//							for (auto v = 0; v < input.size(); ++v) {
//								Vec3b pix = input[v].at<Vec3b>(y + top, x + left);
//								psamples.push_back(Vector3d((double) pix[0], (double) pix[1], (double) pix[2]));
//							}
//						}
//					}
//				}
//				Mat ptrainSample((int)psamples.size(), 3, CV_64F);
//				for(auto i=0; i<psamples.size(); ++i){
//					ptrainSample.at<double>(i,0) = psamples[i][0];
//					ptrainSample.at<double>(i,1) = psamples[i][1];
//					ptrainSample.at<double>(i,2) = psamples[i][2];
//				}
//
//				cout << "Estimating foreground color model..." << endl;
//				gmm_positive->trainEM(ptrainSample);
//
//
//				vector<double> unary;
//				assignColorTerm(input, gmm_positive, gmm_negative, unary);

        }
    }

}//namespace dynamic_stereo

