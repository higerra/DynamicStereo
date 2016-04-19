//
// Created by yanhang on 4/10/16.
//

#include "dynamicsegment.h"
#include "../base/utility.h"
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


	void DynamicSegment::segment(const std::vector<cv::Mat> &warppedImg, cv::Mat &result) const {
		char buffer[1024] = {};

		const int width = warppedImg[0].cols;
		const int height = warppedImg[0].rows;

		result = Mat(height, width, CV_8UC1, Scalar(255));
		vector<Mat> intensityRaw(warppedImg.size());
		for(auto i=0; i<warppedImg.size(); ++i)
			cvtColor(warppedImg[i], intensityRaw[i], CV_BGR2GRAY);

		auto isInside = [&](int x, int y){
			return x>=0 && y >= 0 && x < width && y < height;
		};

		vector<vector<double> > intensity((size_t)width * height);
		for(auto & i: intensity)
			i.resize(warppedImg.size(), 0.0);

		const int pR = 1;
		//box filter with invalid pixel handling
		for(auto i=0; i<warppedImg.size(); ++i) {
			printf("frame %d\n", i+offset);
			for (auto y = 0; y < height; ++y) {
				for (auto x = 0; x < width; ++x) {
					double curI = 0.0;
					double count = 0.0;
					for (auto dx = -1 * pR; dx <= pR; ++dx) {
						for (auto dy = -1 * pR; dy <= pR; ++dy) {
							const int curx = x + dx;
							const int cury = y + dy;
							if(isInside(curx, cury)){
								uchar gv = intensityRaw[i].at<uchar>(cury,curx);
								if(gv == (uchar)0)
									continue;
								count = count + 1.0;
								curI += (double)gv;
							}
						}
					}
					if(count < 1)
						continue;
					intensity[y*width+x][i] = curI / count;
				}
			}
		}

		//brightness confidence dynamicness confidence
		Depth brightness(width, height, 0.0), dynamicness(width, height, 0.0);
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				vector<double>& pixIntensity = intensity[y*width+x];
				CHECK_GT(pixIntensity.size(), 0);
				double count = 0.0;
				//take median as brightness
				const size_t kth = pixIntensity.size() / 2;
				nth_element(pixIntensity.begin(), pixIntensity.begin()+kth, pixIntensity.end());
				brightness(x,y) = pixIntensity[kth];

				double averageIntensity = 0.0;
				for(auto i=0; i<pixIntensity.size(); ++i){
					if(pixIntensity[i] > 0){
						//brightness(x,y) += pixIntensity[i];
						averageIntensity += pixIntensity[i];
						count += 1.0;
					}
				}
				if(count < 2){
					continue;
				}
//				averageIntensity /= count;
//				for(auto i=0; i<pixIntensity.size(); ++i){
//					if(pixIntensity[i] > 0)
//						dynamicness(x,y) += (pixIntensity[i] - averageIntensity) * (pixIntensity[i] - averageIntensity);
//				}
//				if(dynamicness(x,y) > 0)
//					dynamicness(x,y) = std::sqrt(dynamicness(x,y)/(count - 1));
			}
		}

		//compute color difference pattern
		vector<vector<double> > colorDiff((size_t)width*height);
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vec3b refPixv = warppedImg[anchor-offset].at<Vec3b>(y,x);
				Vector3d refPix(refPixv[0], refPixv[1], refPixv[2]);
				double count = 0.0;
				for(auto i=0; i<warppedImg.size(); ++i){
					if(i == anchor-offset)
						continue;
					Vec3b curPixv = warppedImg[i].at<Vec3b>(y,x);
					if(curPixv == Vec3b(0,0,0)) {
						colorDiff[y*width+x].push_back(0.0);
						continue;
					}
					count += 1.0;
					Vector3d curPix(curPixv[0], curPixv[1], curPixv[2]);
					colorDiff[y*width+x].push_back((curPix - refPix).norm());
				}
//				const size_t kth = colorDiff[y*width+x].size()/2;
				//sort(colorDiff[y*width+x].begin(), colorDiff[y*width+x].end(), std::less<double>());
//				nth_element(colorDiff[y*width+x].begin(), colorDiff[y*width+x].begin() + kth, colorDiff[y*width+x].end());
				dynamicness(x,y) = accumulate(colorDiff[y*width+x].begin(), colorDiff[y*width+x].end(), 0.0) / count;

			}
		}

		for(auto i=0; i<warppedImg.size(); ++i){
			sprintf(buffer, "%s/temp/patternb%05d_%05d.txt", file_io.getDirectory().c_str(), anchor, i+offset);
			ofstream fout(buffer);
			CHECK(fout.is_open());
			for(auto y=0; y<height; ++y){
				for(auto x=0; x<width; ++x)
					fout << intensity[y*width+x][i] << ' ';
					//fout << colorDiff[y*width+x][i] << ' ';

				fout << endl;
			}
			fout.close();
		}

		//get array of

		Depth unaryTerm(width, height, 0.0);
		for(auto i=0; i<width * height; ++i)
			unaryTerm[i] = min(dynamicness[i] * brightness[i] / 255.0 * 3, 255.0);

		sprintf(buffer, "%s/temp/conf_brightness%05d.jpg", file_io.getDirectory().c_str(), anchor);
		brightness.saveImage(string(buffer));
		sprintf(buffer, "%s/temp/conf_dynamicness%05d.jpg", file_io.getDirectory().c_str(), anchor);
		dynamicness.saveImage(string(buffer),5);
		sprintf(buffer, "%s/temp/conf_weighted%05d.jpg", file_io.getDirectory().c_str(), anchor);
		unaryTerm.saveImage(string(buffer));

//		unaryTerm.updateStatics();
//		double static_threshold = unaryTerm.getMedianDepth() / 255.0;

		//create problem
		//label 0: don't animate
		//label 1: animate
		printf("Solving by MRF\n");
		std::vector<double> MRF_data((size_t)width*height*2);
		std::vector<double> hCue((size_t)width*height), vCue((size_t)width*height);
		for(auto i=0; i<width*height; ++i){
//			if(unaryTerm[i]/255.0 < static_threshold)
//				MRF_data[2*i] = 0;
//			else
//				MRF_data[2*i] = (unaryTerm[i]/255.0 - 1.0) * (unaryTerm[i]/255.0 - 1.0);
//			MRF_data[2*i] = (unaryTerm[i]/255.0 - 1.0) * (unaryTerm[i]/255.0 - 1.0);
//			MRF_data[2*i+1] = (unaryTerm[i]/255.0) * (unaryTerm[i]/255.0);

			MRF_data[2*i] = unaryTerm[i]/255.0;
			MRF_data[2*i+1] = max(0.0, 0.6 - unaryTerm[i]/255.0);
		}

		const double t = 100;
		const Mat &img = warppedImg[anchor-offset];
		for (auto y = 0; y < height; ++y) {
			for (auto x = 0; x < width; ++x) {
				Vec3b pix1 = img.at<Vec3b>(y, x);
				//pixel value range from 0 to 1, not 255!
				Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]);
				if (y < height - 1) {
					Vec3b pix2 = img.at<Vec3b>(y + 1, x);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					vCue[y*width+x] = std::log(1+std::exp(-1*diff/(t*t)));
				}
				if (x < width - 1) {
					Vec3b pix2 = img.at<Vec3b>(y, x + 1);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					hCue[y*width+x] = std::log(1+std::exp(-1*diff/(t*t)));
				}
			}
		}

		double weight_smooth = 0.5;
		vector<double> MRF_smooth{0,weight_smooth,weight_smooth,0};
		DataCost *dataCost = new DataCost(MRF_data.data());
		SmoothnessCost *smoothnessCost = new SmoothnessCost(MRF_smooth.data(), hCue.data(), vCue.data());
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
