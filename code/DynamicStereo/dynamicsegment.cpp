//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"
#include "../base/utility.h"
#include "dynamic_utility.h"
#include "external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    DynamicSegment::DynamicSegment(const FileIO &file_io_, const int anchor_, const int tWindow_, const int downsample_,
    const std::vector<Depth>& depths_, const std::vector<int>& depthInd_):
            file_io(file_io_), anchor(anchor_), downsample(downsample_) {
        sfmModel.init(file_io.getReconstruction());

	    //load image
	    if (anchor - tWindow_ / 2 < 0) {
		    offset = 0;
		    CHECK_LT(offset + tWindow_, file_io.getTotalNum());
	    } else if (anchor + tWindow_ / 2 >= file_io.getTotalNum()) {
		    offset = file_io.getTotalNum() - 1 - tWindow_;
		    CHECK_GE(offset, 0);
	    } else
		    offset = anchor - tWindow_ / 2;

//	    images.resize((size_t) tWindow_);
//	    for (auto i = 0; i < images.size(); ++i) {
//		    images[i] = imread(file_io.getImage(i + offset));
//		    for (auto y = 0; y < images[i].rows; ++y) {
//			    for (auto x = 0; x < images[i].cols; ++x) {
//				    if (images[i].at<Vec3b>(y, x) == Vec3b(0, 0, 0))
//					    images[i].at<Vec3b>(y, x) = Vec3b(1, 1, 1);
//			    }
//		    }
//	    }

//	    CHECK(!images.empty());
//	    width = images[0].cols;
//	    height = images[0].rows;
//
//	    CHECK_EQ(depths.size(), depthInd.size());
//	    for(auto i=0; i<depthInd.size(); ++i){
//			depths[i].updateStatics();
//		    if(depthInd[i] == anchor){
//			    refDepth = depths[i];
//		    }
//	    }
//
//	    CHECK_EQ(refDepth.getWidth(), width / downsample);
//	    CHECK_EQ(refDepth.getHeight(), height / downsample);
    }

//	void DynamicSegment::getGeometryConfidence(Depth &geoConf) const {
//		geoConf.initialize(width, height, 0.0);
//		const theia::Camera& refCam = sfmModel.getCamera(anchor);
//		const double large = 1000000;
//
//		for(auto y=downsample; y<height-downsample; ++y){
//			for(auto x=downsample; x<width-downsample; ++x){
//				Vector2d refPt((double)x/(double)downsample, (double)y/(double)downsample);
//				Vector3d spt = refCam.GetPosition() + refCam.PixelToUnitDepthRay(Vector2d(x,y)) * refDepth.getDepthAt(refPt);
//				vector<double> repoError;
//
//				for(auto j=0; j<depthInd.size(); ++j){
//					Vector2d imgpt;
//					const theia::Camera& cam2 = sfmModel.getCamera(depthInd[j]);
//					double d = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
//					imgpt /= (double)downsample;
//					if(d > 0 && imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < depths[j].getWidth()-1 && imgpt[1] < depths[j].getHeight()-1){
//						double depth2 = depths[j].getDepthAt(imgpt);
//						double zMargin = depths[j].getMedianDepth() / 5;
//						if(d <= depth2 + zMargin) {
//							Vector3d spt2 = cam2.PixelToUnitDepthRay(imgpt * downsample) * depth2 + cam2.GetPosition();
//							Vector2d repoPt;
//							double repoDepth = refCam.ProjectPoint(spt2.homogeneous(), &repoPt);
//							double dis = (repoPt - Vector2d(x,y)).norm();
//							repoError.push_back(dis);
//						}
//					}
//				}
//
//				//take average
//				if(!repoError.empty())
//					geoConf(x,y) = std::accumulate(repoError.begin(), repoError.end(), 0.0) / (double)repoError.size();
//			}
//		}
//	}


	void DynamicSegment::computeFrequencyConfidence(const std::vector<cv::Mat> &warppedImg, Depth &result) const {
		CHECK(!warppedImg.empty());
		const int width = warppedImg[0].cols;
		const int height = warppedImg[0].rows;
		result.initialize(width, height, 0.0);
		const int N = (int)warppedImg.size();
		const double alpha = 2, beta = 2.0;
		const float epsilon = 1e-05;
		const int min_frq = 3;
		const int tx = -1, ty= -1;

		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				//seprate first half and second half
				vector<Vector3f> meanColor(2, Vector3f(0,0,0));
				Mat colorArray = Mat(3,N,CV_32FC1, Scalar::all(0));
				float* pArray = (float*)colorArray.data;
				for(auto v=0; v<warppedImg.size(); ++v){
					Vec3b pixv = warppedImg[v].at<Vec3b>(y,x);
					int ind = v < N/2 ? 0 : 1;
					if(pixv[0] != 0 && pixv[1] != 0 && pixv[2] != 0){
						pArray[v] = (float)pixv[0];
						pArray[N+v] = (float)pixv[1];
						pArray[2*N+v] = (float)pixv[2];
					}
					meanColor[ind][0] += pArray[v];
					meanColor[ind][1] += pArray[N+v];
					meanColor[ind][2] += pArray[2*N+v];
				}
				for(auto h=0; h<2; ++h) {
					meanColor[h] /= (double) N / 2.0;
				}
				for (auto v = 0; v < N; ++v) {
					int ind = v < N / 2 ? 0 : 1;
					pArray[v] -= meanColor[ind][0];
					pArray[N + v] -= meanColor[ind][1];
					pArray[2 * N + v] -= meanColor[ind][2];
				}

				vector<double> frqConfs{utility::getFrequencyScore(colorArray(cv::Range::all(), cv::Range(0,N/2)), min_frq),
										utility::getFrequencyScore(colorArray(cv::Range::all(), cv::Range(N/2,N)), min_frq)};
				result(x,y) = 1 / (1 + std::exp(-1*alpha*(std::max(frqConfs[0], frqConfs[1]) - beta)));
				if(x == tx && y == ty){
					for(auto v=0; v<N/2; ++v){
						printf("%.2f,%.2f,%.2f\n", pArray[v], pArray[N+v], pArray[2*N+v]);
					}
					printf("mean1: %.2f,%.2f,%.2f\n", meanColor[0][0], meanColor[0][1], meanColor[0][2]);
					printf("mean2: %.2f,%.2f,%.2f\n", meanColor[1][0], meanColor[1][1], meanColor[1][2]);
					printf("(%d,%d),frqConf:[%.2f,%.2f], result:%.2f\n", tx,ty,frqConfs[0], frqConfs[1], result(x,y));
				}
			}
		}

	}


	void DynamicSegment::segment(const std::vector<cv::Mat> &warppedImg, cv::Mat &result) const {
		char buffer[1024] = {};

		const int width = warppedImg[0].cols;
		const int height = warppedImg[0].rows;

		result = Mat(height, width, CV_8UC1, Scalar(0));
		uchar* pResult = result.data;
//		vector<Mat> intensityRaw(warppedImg.size());
//		for(auto i=0; i<warppedImg.size(); ++i)
//			cvtColor(warppedImg[i], intensityRaw[i], CV_BGR2GRAY);
//
//		auto isInside = [&](int x, int y){
//			return x>=0 && y >= 0 && x < width && y < height;
//		};
//
//		vector<vector<double> > intensity((size_t)width * height);
//		for(auto & i: intensity)
//			i.resize(warppedImg.size(), 0.0);
//
//		const int pR = 0;
//
//		//box filter with invalid pixel handling
//		for(auto i=0; i<warppedImg.size(); ++i) {
//			printf("frame %d\n", i+offset);
//			for (auto y = 0; y < height; ++y) {
//				for (auto x = 0; x < width; ++x) {
//					double curI = 0.0;
//					double count = 0.0;
//					for (auto dx = -1 * pR; dx <= pR; ++dx) {
//						for (auto dy = -1 * pR; dy <= pR; ++dy) {
//							const int curx = x + dx;
//							const int cury = y + dy;
//							if(isInside(curx, cury)){
//								uchar gv = intensityRaw[i].at<uchar>(cury,curx);
//								if(gv == (uchar)0)
//									continue;
//								count = count + 1.0;
//								curI += (double)gv;
//							}
//						}
//					}
//					if(count < 1)
//						continue;
//					intensity[y*width+x][i] = curI / count;
//				}
//			}
//		}
//
////		for(auto i=0; i<warppedImg.size(); ++i){
////			sprintf(buffer, "%s/temp/patternb%05d_%05d.txt", file_io.getDirectory().c_str(), anchor, i+offset);
////			ofstream fout(buffer);
////			CHECK(fout.is_open());
////			for(auto y=0; y<height; ++y){
////				for(auto x=0; x<width; ++x)
////					fout << intensity[y*width+x][i] << ' ';
////				//fout << colorDiff[y*width+x][i] << ' ';
////				fout << endl;
////			}
////			fout.close();
////		}
//
//
//		//brightness confidence dynamicness confidence
//		Depth brightness(width, height, 0.0), dynamicness(width, height, 0.0);
//		for(auto y=0; y<height; ++y){
//			for(auto x=0; x<width; ++x){
//				vector<double>& pixIntensity = intensity[y*width+x];
//				CHECK_GT(pixIntensity.size(), 0);
//				double count = 0.0;
//				//take median as brightness
//				const size_t kth = pixIntensity.size() / 2;
//				nth_element(pixIntensity.begin(), pixIntensity.begin()+kth, pixIntensity.end());
//				brightness(x,y) = pixIntensity[kth];
//
//				double averageIntensity = 0.0;
//				for(auto i=0; i<pixIntensity.size(); ++i){
//					if(pixIntensity[i] > 0){
//						//brightness(x,y) += pixIntensity[i];
//						averageIntensity += pixIntensity[i];
//						count += 1.0;
//					}
//				}
//				if(count < 2){
//					continue;
//				}
////				averageIntensity /= count;
////				for(auto i=0; i<pixIntensity.size(); ++i){
////					if(pixIntensity[i] > 0)
////						dynamicness(x,y) += (pixIntensity[i] - averageIntensity) * (pixIntensity[i] - averageIntensity);
////				}
////				if(dynamicness(x,y) > 0)
////					dynamicness(x,y) = std::sqrt(dynamicness(x,y)/(count - 1));
//			}
//		}
//
//		//compute color difference pattern
//		vector<vector<double> > colorDiff((size_t)width*height);
//		for(auto y=0; y<height; ++y){
//			for(auto x=0; x<width; ++x){
//				Vec3b refPixv = warppedImg[anchor-offset].at<Vec3b>(y,x);
//				Vector3d refPix(refPixv[0], refPixv[1], refPixv[2]);
//				double count = 0.0;
//				for(auto i=0; i<warppedImg.size(); ++i){
//					if(i == anchor-offset)
//						continue;
//					Vec3b curPixv = warppedImg[i].at<Vec3b>(y,x);
//					if(curPixv == Vec3b(0,0,0)) {
//						colorDiff[y*width+x].push_back(0.0);
//						continue;
//					}
//					count += 1.0;
//					Vector3d curPix(curPixv[0], curPixv[1], curPixv[2]);
//					colorDiff[y*width+x].push_back((curPix - refPix).norm());
//				}
//				if(count < 1)
//					continue;
////				const size_t kth = colorDiff[y*width+x].size()/2;
////				sort(colorDiff[y*width+x].begin(), colorDiff[y*width+x].end(), std::less<double>());
////				nth_element(colorDiff[y*width+x].begin(), colorDiff[y*width+x].begin() + kth, colorDiff[y*width+x].end());
//				dynamicness(x,y) = accumulate(colorDiff[y*width+x].begin(), colorDiff[y*width+x].end(), 0.0) / count;
//			}
//		}

		//repetative pattern
		printf("Computing frequency confidence...\n");
		Depth frequency;
		computeFrequencyConfidence(warppedImg, frequency);
		printf("Done\n");


		//compute anisotropic weight
		const double t = 80;
		const double min_diffusion = 0.15;
		const cv::Mat& img = warppedImg[anchor-offset];
		vector<double> hCue((size_t)width * height), vCue((size_t)width * height);
		for (auto y = 0; y < height; ++y) {
			for (auto x = 0; x < width; ++x) {
				Vec3b pix1 = img.at<Vec3b>(y, x);
				//pixel value range from 0 to 1, not 255!
				Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]);
				if (y < height - 1) {
					Vec3b pix2 = img.at<Vec3b>(y + 1, x);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					vCue[y*width+x] = std::max(std::log(1+std::exp(-1*diff/(t*t))), min_diffusion);
				}
				if (x < width - 1) {
					Vec3b pix2 = img.at<Vec3b>(y, x + 1);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					hCue[y*width+x] = std::max(std::log(1+std::exp(-1*diff/(t*t))), min_diffusion);
				}
			}
		}


		{
			//test for grabuct segmentation
			printf("Segmentation based on frequency...\n");
			Mat bwmask(height, width, CV_8UC1, Scalar::all(0));
			Mat bgmask(height, width, CV_8UC1, Scalar::all(0));
			uchar *pBwmask = bwmask.data;
			uchar *pBgmask = bgmask.data;

			//initial mask
			const double tl = 0.1, th = 0.5;
			for (auto i = 0; i < width * height; ++i)
				pBwmask[i] = frequency[i] > th ? (uchar) 255 : (uchar) 0;
			for (auto i = 0; i < width * height; ++i)
				pBgmask[i] = frequency[i] < tl ? (uchar) 255 : (uchar) 0;
			const int rh = 5;
			const int min_area = 150;

			cv::dilate(bwmask, bwmask, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rh, rh)));
			cv::erode(bwmask, bwmask, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(rh, rh)));
			cv::erode(bgmask, bgmask, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(11, 11)));

			sprintf(buffer, "%s/temp/mask_frqfg%05d.jpg", file_io.getDirectory().c_str(), anchor);
			imwrite(buffer, bwmask);
			sprintf(buffer, "%s/temp/mask_frqbg%05d.jpg", file_io.getDirectory().c_str(), anchor);
			imwrite(buffer, bgmask);

			sprintf(buffer, "%s/midres/gmm_negative%05d.gmm", file_io.getDirectory().c_str(), anchor);
			cv::Ptr<cv::ml::EM> gmm_negative = cv::ml::EM::create();


			//collect negative sample
			vector<Vector3d> nsamples;
			for(auto y=0; y<height; ++y){
				for(auto x=0; x<width; ++x){
					if(pBgmask[y*width+x] > 200){
						for(auto v=0; v<warppedImg.size(); ++v){
							Vec3b pix = warppedImg[v].at<Vec3b>(y,x);
							nsamples.push_back(Vector3d((double)pix[0], (double)pix[1], (double)pix[2]));
						}
					}
				}
			}
			Mat ntrainSample((int)nsamples.size(), 3, CV_64F);
			for(auto i=0; i<nsamples.size(); ++i){
				ntrainSample.at<double>(i,0) = nsamples[i][0];
				ntrainSample.at<double>(i,1) = nsamples[i][1];
				ntrainSample.at<double>(i,2) = nsamples[i][2];
			}
			cout << "Estimating background color model..." << endl;
			gmm_negative->trainEM(ntrainSample);


			//connected component analysis
			Mat labels, stats, centroids;
			int nCom = cv::connectedComponentsWithStats(bwmask, labels, stats, centroids);
			const int *pLabel = (int *) labels.data;
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

//				const int left = stats.at<int>(l, CC_STAT_LEFT);
//				const int top = stats.at<int>(l, CC_STAT_TOP);
//				const int roiw = stats.at<int>(l, CC_STAT_WIDTH);
//				const int roih = stats.at<int>(l, CC_STAT_HEIGHT);
				const int left = 0;
				const int top = 0;
				const int roiw = width;
				const int roih = height;

//				Mat roi = warppedImg[anchor-offset](cv::Rect(left, top, roiw, roih));
//
//				Mat gcmask(roih, roiw, CV_8UC1, Scalar::all(GC_PR_BGD));
//				uchar *pGcmask = gcmask.data;
//				for(auto y=0; y<roih; ++y){
//					for(auto x=0; x<roiw; ++x){
//						int oriId = (y+top) * width + x + left;
//						if(pLabel[oriId] == l)
//							pGcmask[y*roiw + x] = GC_FGD;
//						else if(pBgmask[oriId] > 200)
//							pGcmask[y*roiw + x] = GC_BGD;
//					}
//				}
//				printf("Grabcut...\n");
//				grabCut(roi, gcmask, cv::Rect(), cv::Mat(), cv::Mat(), GC_INIT_WITH_MASK);
//
//				for(auto y=0; y<roih; ++y){
//					for(auto x=0; x<roiw; ++x){
//						int oriId = (y+top) * width + x + left;
//						if(pGcmask[y*roiw+x] == GC_FGD || pGcmask[y*roiw+x] == GC_PR_FGD)
//							pResult[oriId] = (uchar)255;
//					}
//				}


				//estimate GMM
				cv::Ptr<cv::ml::EM> gmm_positive = cv::ml::EM::create();

				vector<Vector3d> psamples;
				//collect training sample
				for (auto y = top; y < top + roih; ++y) {
					for (auto x = left; x < left + roiw; ++x) {
						if (pLabel[(y + top) * width + x + left] == l) {
							for (auto v = 0; v < warppedImg.size(); ++v) {
								Vec3b pix = warppedImg[v].at<Vec3b>(y + top, x + left);
								psamples.push_back(Vector3d((double) pix[0], (double) pix[1], (double) pix[2]));
							}
						}
					}
				}
				Mat ptrainSample((int)psamples.size(), 3, CV_64F);
				for(auto i=0; i<psamples.size(); ++i){
					ptrainSample.at<double>(i,0) = psamples[i][0];
					ptrainSample.at<double>(i,1) = psamples[i][1];
					ptrainSample.at<double>(i,2) = psamples[i][2];
				}

				cout << "Estimating foreground color model..." << endl;
				gmm_positive->trainEM(ptrainSample);


				vector<double> unary;
				assignColorTerm(warppedImg, gmm_positive, gmm_negative, unary);

			}
		}


//		Depth unaryTerm(width, height, 0.0);
//		for(auto i=0; i<width * height; ++i)
//			unaryTerm[i] = min(dynamicness[i] * brightness[i] / 255.0 * 3, 255.0);
//
//		sprintf(buffer, "%s/temp/conf_brightness%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		brightness.saveImage(string(buffer));
//		sprintf(buffer, "%s/temp/conf_dynamicness%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		dynamicness.saveImage(string(buffer),5);
//		sprintf(buffer, "%s/temp/conf_weighted%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		unaryTerm.saveImage(string(buffer));
		sprintf(buffer, "%s/temp/conf_frquency%05d.jpg", file_io.getDirectory().c_str(), anchor);
		frequency.saveImage(string(buffer), 255);

//		unaryTerm.updateStatics();
//		double static_threshold = unaryTerm.getMedianDepth() / 255.0;

		//create problem
		//label 0: don't animate
		//label 1: animate
//		printf("Solving by MRF\n");
//		std::vector<double> MRF_data((size_t)width*height*2);

	}

	void DynamicSegment::assignColorTerm(const std::vector<cv::Mat> &warped, const Ptr<cv::ml::EM> fgModel,
										 const cv::Ptr<cv::ml::EM> bgModel, std::vector<double> &colorTerm)const {
		CHECK(!warped.empty());
		const int width = warped[0].cols;
		const int height = warped[0].rows;
		colorTerm.resize((size_t)width * height * 2);
		for(auto v=0; v<warped.size(); ++v){
			const uchar* pImg = warped[v].data;
			for(auto i=0; i<width * height; ++i){
				Mat x(3,1,CV_64F);
				double* pX = (double*) x.data;
				pX[0] = pImg[3*i];
				pX[1] = pImg[3*i+1];
				pX[2] = pImg[3*i+2];
				Vec2d predfg = fgModel->predict2(x, Mat());
				Vec2d predbg = bgModel->predict2(x, Mat());
				//use negative log likelihood for energy
				colorTerm[2*i] = -1 * predbg[0];
				colorTerm[2*i+1] = -1 * predfg[0];
			}
		}
	}

	void DynamicSegment::solveMRF(const std::vector<double> &unary,
								  const std::vector<double>& vCue,
								  const std::vector<double>& hCue,
								  const cv::Mat& img, const double weight_smooth,
								  cv::Mat& result) const {
		const int width = img.cols;
		const int height = img.rows;

		double MRF_smooth[] = {0,weight_smooth,weight_smooth,0};

		DataCost *dataCost = new DataCost(const_cast<double*>(unary.data()));
		SmoothnessCost *smoothnessCost = new SmoothnessCost(MRF_smooth, const_cast<double*>(hCue.data()), const_cast<double*>(vCue.data()));
		EnergyFunction* energy_function = new EnergyFunction(dataCost, smoothnessCost);

		Expansion mrf(width,height,2,energy_function);
		mrf.initialize();
		mrf.clearAnswer();
		for(auto i=0; i<width*height; ++i)
			mrf.setLabel(i,0);
		double initDataE = mrf.dataEnergy();
		double initSmoothE = mrf.smoothnessEnergy();
		float mrf_time;
		mrf.optimize(10, mrf_time);
		printf("Inital energy: (%.3f,%.3f,%.3f), final energy: (%.3f,%.3f,%.3f), time:%.2fs\n", initDataE, initSmoothE, initDataE+initSmoothE,
		       mrf.dataEnergy(), mrf.smoothnessEnergy(), mrf.dataEnergy()+mrf.smoothnessEnergy(), mrf_time);

		result = Mat(height, width, CV_8UC1);
		uchar* pResult = result.data;
		for(auto i=0; i<width*height; ++i){
			if(mrf.getLabel(i) > 0)
				pResult[i] = (uchar)255;
			else
				pResult[i] = (uchar)0;
		}
		delete dataCost;
		delete smoothnessCost;
		delete energy_function;

	}

}//namespace dynamic_stereo
