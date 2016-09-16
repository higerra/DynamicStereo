//
// Created by yanhang on 5/4/16.
//
#include "factorization.h"
using namespace cv;
using namespace std;
using namespace Eigen;

namespace substab{
    namespace Factorization {

	static void factorizeWindow(const FeatureTracks& trackMatrix, Eigen::MatrixXd& coe, Eigen::MatrixXd& bas,
				    vector<vector<int> >& wMatrix, std::vector<bool>& is_computed,
				    const int tWindow, const int stride, const int v){
	    const int kTrack = (int) trackMatrix.tracks.size();
	    const int kBasis = 9;
	    
	    vector<int> preFullTrackInd;
	    vector<int> newFullTrackInd;
	    for (auto tid = 0; tid < kTrack; ++tid) {
		if (trackMatrix.offset[tid] <= v &&
		    trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() >= v + tWindow) {
		    bool is_valid = true;
		    for(auto i=v; i<v+tWindow; ++i){
			if(wMatrix[tid][i] == -1) {
			    is_valid = false;
			    break;
			}
		    }
		    if(!is_valid)
			continue;
		    if (trackMatrix.offset[tid] <= v - stride) {
			//complete track inside previous window
			CHECK(is_computed[tid]);
			preFullTrackInd.push_back(tid);
		    } else {
			//new complete track
			newFullTrackInd.push_back(tid);
		    }
		}
	    }

//				filterDynamicTrack(images, trackMatrix, preFullTrackInd, v, tWindow, wMatrix);
//				filterDynamicTrack(images, trackMatrix, newFullTrackInd, v, tWindow, wMatrix);

	    MatrixXd A12 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, stride);
	    MatrixXd A2 = MatrixXd::Zero((int) newFullTrackInd.size() * 2, tWindow);
	    MatrixXd C1 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, kBasis);
	    MatrixXd A11 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, tWindow - stride);

	    MatrixXd E1 = bas.block(0, v, bas.rows(), tWindow - stride);

	    for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
		const int idx = preFullTrackInd[ftid];
		const int offset = (int) trackMatrix.offset[idx];
		CHECK(is_computed[idx]);
		for (auto i = v + tWindow - stride; i < v + tWindow; ++i) {
		    CHECK_GE(i-offset, 0);
		    CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
		    A12(ftid * 2, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].x;
		    A12(ftid * 2 + 1, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].y;
		}
		C1.block(ftid*2, 0, 2, C1.cols()) = coe.block(idx*2,0,2,coe.cols());
	    }

	    for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
		const int idx = newFullTrackInd[ftid];
		const int offset = (int) trackMatrix.offset[idx];
		for (auto i = v; i < v + tWindow; ++i) {
		    CHECK_GE(i-offset, 0);
		    CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
		    A2(ftid * 2, i - v) = trackMatrix.tracks[idx][i - offset].x;
		    A2(ftid * 2 + 1, i - v) = trackMatrix.tracks[idx][i - offset].y;
		}
	    }

	    MatrixXd A21 = A2.block(0, 0, A2.rows(), tWindow - stride);
	    MatrixXd A22 = A2.block(0, tWindow - stride, A2.rows(), stride);

	    MatrixXd EE = E1 * E1.transpose();
	    MatrixXd EEinv = EE.inverse();
	    MatrixXd C2 = A21 * E1.transpose() * EEinv;

	    MatrixXd largeC(C1.rows() + C2.rows(), C1.cols());
	    largeC.block(0,0,C1.rows(),kBasis) = C1;
	    largeC.block(C1.rows(),0,C2.rows(), kBasis) = C2;

	    MatrixXd largeA(A12.rows() + A22.rows(), A12.cols());
	    largeA.block(0,0,A12.rows(),A12.cols()) = A12;
	    largeA.block(A12.rows(),0,A22.rows(), A22.cols()) = A22;

	    MatrixXd E2 = (largeC.transpose() * largeC).inverse() * largeC.transpose() * largeA;
	    //MatrixXd E2 = (C1.transpose() * C1).inverse() * C1.transpose() * A12;
	    //MatrixXd E2 = (C2.transpose() * C2).inverse() * C2.transpose() * A22;

	    bas.block(0, v + tWindow - stride, bas.rows(), stride) = E2;

	    for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
		const int idx = newFullTrackInd[ftid];
		coe.block(2 * idx, 0, 2, coe.cols()) = C2.block(2 * ftid, 0, 2, C2.cols());
		for(auto i=v; i< v+ tWindow; ++i)
		    wMatrix[idx][i] = 1;
		is_computed[idx] = true;
	    }
	}
	    
	void movingFactorization(const vector<Mat> &images, const FeatureTracks &trackMatrix, Eigen::MatrixXd &coe,
				 Eigen::MatrixXd &bas, vector<vector<int> >& wMatrix, const int tWindow, const int stride) {
	    const int kBasis = 9;
	    const int kTrack = (int) trackMatrix.tracks.size();
	    char buffer[1024] = {};
	    const int N = (int)images.size();

	    vector<bool> is_computed(kTrack, false);

	    coe = MatrixXd::Zero(2 * kTrack, kBasis);
	    bas = MatrixXd::Zero(kBasis, N);

	    const int testV = 0;
	    //factorize the first window
	    {
		vector<int> fullTrackInd;
		for (auto tid = 0; tid < kTrack; ++tid) {
		    if (trackMatrix.offset[tid] <= testV) {
			if (trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() >= tWindow + testV)
			    fullTrackInd.push_back(tid);
		    }
		}


//				filterDynamicTrack(images, trackMatrix, fullTrackInd, testV, tWindow, wMatrix);
		MatrixXd A((int) fullTrackInd.size() * 2, tWindow);

		for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
		    const int idx = fullTrackInd[ftid];
		    const int offset = (int) trackMatrix.offset[idx];
		    for (auto i = testV; i < tWindow + testV; ++i) {
			A(ftid * 2, i - testV) = trackMatrix.tracks[idx][i - offset].x;
			A(ftid * 2 + 1, i - testV) = trackMatrix.tracks[idx][i - offset].y;
		    }
		}

		Eigen::JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
		MatrixXd w = svd.singularValues().block(0, 0, kBasis, 1).asDiagonal();
		w = w.array().sqrt();
		MatrixXd curcoe = svd.matrixU().block(0, 0, A.rows(), kBasis) * w;
		MatrixXd curbas = w * svd.matrixV().transpose().block(0, 0, kBasis, A.cols());

		bas.block(0, testV, kBasis, tWindow) = curbas;
		for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
		    const int idx = fullTrackInd[ftid];
		    coe.block(2 * idx, 0, 2, kBasis) = curcoe.block(2 * ftid, 0, 2, kBasis);
		    for(auto i=testV; i<testV+tWindow; ++i)
			wMatrix[idx][i] = 1;
		    is_computed[idx] = true;
		}
	    }

	    //moving factorization
	    for (auto v = stride + testV; v < N - tWindow; v += stride) {
		factorizeWindow(trackMatrix, coe, bas, wMatrix, is_computed, tWindow, stride, v);
	    }

	    //factorize last window
	    factorizeWindow(trackMatrix, coe, bas, wMatrix, is_computed, tWindow, stride, N - tWindow);
	}


	void filterDynamicTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix, std::vector<int>& fullTrackInd,
				const int sf, const int tw, vector<vector<int> >& wMatrix){
	    const double max_ratio = 0.33;
	    vector<double> totalCount(fullTrackInd.size(), 0.0);
	    vector<double> outlierCount(fullTrackInd.size(), 0.0);
	    const double max_epiError = 2.0;
	    const int stride = 5;

	    for(auto v=0; v<sf+tw-stride; v+=stride) {
		vector<cv::Point2f> pts1, pts2;
		vector<size_t> trackId;
		for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
		    const int idx = fullTrackInd[ftid];
		    const int offset = (int) trackMatrix.offset[idx];
		    if(offset <= v && offset+trackMatrix.tracks[idx].size() >= v+stride) {
			pts1.push_back(trackMatrix.tracks[idx][v]);
			pts2.push_back(trackMatrix.tracks[idx][v+stride-1]);
			trackId.push_back(ftid);
		    }
		}
		if(trackId.size() < 8)
		    continue;
		Mat fMatrix = cv::findFundamentalMat(pts1, pts2);
		if(fMatrix.cols != 3)
		    continue;
		Mat epilines;
		cv::computeCorrespondEpilines(pts1, 1, fMatrix, epilines);
		for(auto ptid=0; ptid<trackId.size(); ++ptid){
		    Vector3d epi(epilines.at<Vec3f>(ptid,0)[0], epilines.at<Vec3f>(ptid,0)[1], epilines.at<Vec3f>(ptid,0)[2]);
		    Vector3d pt(pts2[ptid].x, pts2[ptid].y, 1.0);
		    totalCount[trackId[ptid]] += 1.0;
		    if(epi.dot(pt) > max_epiError)
			outlierCount[trackId[ptid]] += 1.0;
		}
	    }

	    vector<int> inlierInd;

	    for(auto i=0; i<fullTrackInd.size(); ++i){
		if(outlierCount[i] / max_ratio <= totalCount[i])
		    inlierInd.push_back(fullTrackInd[i]);
		else{
		    for(auto v=sf; v<wMatrix[fullTrackInd[i]].size(); ++v)
			wMatrix[fullTrackInd[i]][v] = -1;
		}
	    }
	    fullTrackInd.swap(inlierInd);
	}

	void trackSmoothing(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, const int r, const double sigma){
	    Mat gauKernelCV = cv::getGaussianKernel(2*r+1,sigma,CV_64F);
	    const double* pKernel = (double*) gauKernelCV.data;
	    output = MatrixXd::Zero(input.rows(), input.cols());
	    for(auto i=0; i<input.rows(); ++i){
		for(auto j=0; j<input.cols(); ++j){
		    double sum = 0.0;
		    for(auto k=-1*r; k<=r; ++k){
			if(j+k < 0 || j+k >= input.cols())
			    continue;
			sum += pKernel[k+r];
			output(i,j) += input(i,j+k) * pKernel[k+r];
		    }
		    output(i,j) /= sum;
		}
	    }
	}

    }//namespace Factorization
}//namespace substab
