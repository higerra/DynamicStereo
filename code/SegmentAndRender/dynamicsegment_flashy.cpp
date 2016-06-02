//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"
#include "../base/utility.h"
#include "../common/dynamic_utility.h"
#include "external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

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


	void computeFrequencyConfidence(const std::vector<cv::Mat> &warppedImg, Depth &result){
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


	void segmentFlashy(const FileIO& file_io, const int anchor,
	                   const std::vector<cv::Mat> &input, cv::Mat &result){
		char buffer[1024] = {};

		const int width = input[0].cols;
		const int height = input[0].rows;

		result = Mat(height, width, CV_8UC1, Scalar(0));
		uchar* pResult = result.data;

		//repetative pattern
		printf("Computing frequency confidence...\n");
		Depth frequency;
		computeFrequencyConfidence(input, frequency);
		printf("Done\n");


		//compute anisotropic weight
//		const double t = 80;
//		const double min_diffusion = 0.15;
//		const cv::Mat& img = input[anchor-offset];
//		vector<double> hCue((size_t)width * height), vCue((size_t)width * height);
//		for (auto y = 0; y < height; ++y) {
//			for (auto x = 0; x < width; ++x) {
//				Vec3b pix1 = img.at<Vec3b>(y, x);
//				//pixel value range from 0 to 1, not 255!
//				Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]);
//				if (y < height - 1) {
//					Vec3b pix2 = img.at<Vec3b>(y + 1, x);
//					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
//					double diff = (dpix1 - dpix2).squaredNorm();
//					vCue[y*width+x] = std::max(std::log(1+std::exp(-1*diff/(t*t))), min_diffusion);
//				}
//				if (x < width - 1) {
//					Vec3b pix2 = img.at<Vec3b>(y, x + 1);
//					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
//					double diff = (dpix1 - dpix2).squaredNorm();
//					hCue[y*width+x] = std::max(std::log(1+std::exp(-1*diff/(t*t))), min_diffusion);
//				}
//			}
//		}


		{

		}


//		Depth unaryTerm(width, height, 0.0);
//		for(auto i=0; i<width * height; ++i)
//			unaryTerm[i] = min(dynamicness[i] * brightness[i] / 255.0 * 3, 255.0);
//
//		sprintf(buffer, "%s/temp/conf_brightness%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		brightness.saveImage(string(buffer));
//		sprintf(buffer, "%s/temp/conf_weighted%05d.jpg", file_io.getDirectory().c_str(), anchor);
//		unaryTerm.saveImage(string(buffer));
		sprintf(buffer, "%s/temp/conf_frquency.jpg", file_io.getDirectory().c_str());
		frequency.saveImage(string(buffer), 255);

//		unaryTerm.updateStatics();
//		double static_threshold = unaryTerm.getMedianDepth() / 255.0;

		//create problem
		//label 0: don't animate
		//label 1: animate
//		printf("Solving by MRF\n");
//		std::vector<double> MRF_data((size_t)width*height*2);

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
