#include "depth.h"
#include <algorithm>
#include <assert.h>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <glog/logging.h>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {

	void Depth::initialize(int width, int height, const double v) {
		depthwidth = width;
		depthheight = height;
		for (int i = 0; i < depthwidth * depthheight; i++) {
			data.push_back(v);
			if (v >= 0)
				weight.push_back(1);
			else
				weight.push_back(0);
		}

	}

	double Depth::getDepthAt(const Vector2d &loc) const {
		double x = loc[0], y = loc[1];
		int xl = floor(x), xh = round(x + 0.5);
		int yl = floor(y), yh = round(y + 0.5);
		double lm = x - (double) xl, rm = (double) xh - x;
		double tm = y - (double) yl, bm = (double) yh - y;
		Vector4d ind(xl + yl * getWidth(), xh + yl * getWidth(), xh + yh * getWidth(), xl + yh * getWidth());
		Vector4d w(rm * bm, lm * bm, lm * tm, rm * tm);
		Vector4d v(data[ind[0]], data[ind[1]], data[ind[2]], data[ind[3]]);
		for (int i = 0; i < 4; i++) {
			if (v[i] == -1)
				w[i] = 0;
		}

		if (w[0] + w[1] + w[2] + w[3] == 0)
			return -10;
		return w.dot(v);
	}

	void Depth::setDepthAt(const Vector2d &pix, double v) {
		const double max_diff = 1.5;
		int xl = floor(pix[0]), xh = round(pix[0] + 0.5);
		int yl = floor(pix[1]), yh = round(pix[1] + 0.5);
		double lm = pix[0] - (double) xl, rm = (double) xh - pix[0];
		double tm = pix[1] - (double) yl, bm = (double) yh - pix[1];
		Vector4d w(rm * bm, lm * bm, lm * tm, rm * tm);
		Vector4d w_ori(getWeightAt(xl, yl), getWeightAt(xh, yl), getWeightAt(xh, yh), getWeightAt(xl, yh));
		Vector4d ind(xl + yl * getWidth(), xh + yl * getWidth(), xh + yh * getWidth(), xl + yh * getWidth());
		for (int i = 0; i < 4; i++) {
			if ((data[ind[i]] < 0 || abs(data[ind[i]] - v) < max_diff) && w_ori[i] + w[i] != 0) {
				data[ind[i]] = (data[ind[i]] * w_ori[i] + v * w[i]) / (w_ori[i] + w[i]);
				setWeightAt(ind[i], w_ori[i] + w[i]);
			}
		}
	}

	void Depth::saveImage(const string &filename, const double amp) const {
		Mat outmat(getHeight(), getWidth(), CV_8UC3);
		uchar* pData = outmat.data;
		for (int y = 0; y < getHeight(); y++) {
			for (int x = 0; x < getWidth(); x++) {
				const int idx = x + y * getWidth();
				double curdepth = data[idx];
				double dv = curdepth * amp;
				if(dv > 255)
					dv = 255;
				if (curdepth >= 0)
					for(int i=0; i<3; ++i)
						pData[3*idx+i] = (uchar)dv;
				else{
					pData[3*idx] = 255;
					pData[3*idx+1] = 0;
					pData[3*idx+2] = 0;
				}
			}
		}
		imwrite(filename, outmat);
	}

	bool Depth::readDepthFromFile(const string &filename) {
		ifstream fin(filename.c_str(), ios::binary);
		if (!fin.is_open()) {
			fin.close();
			return false;
		}
		fin.read((char *) &depthwidth, sizeof(int));
		fin.read((char *) &depthheight, sizeof(int));
		data.resize(depthwidth * depthheight);
		fin.read((char *) &data[0], data.size() * sizeof(double));
		fin.close();
		updateStatics();
		return true;
	}

	void Depth::saveDepthFile(const string &filename) const {
		ofstream depthout(filename.c_str(), ios::binary);
		int width = getWidth();
		int height = getHeight();
		depthout.write((char *) &width, sizeof(int));
		depthout.write((char *) &height, sizeof(int));
		depthout.write((char *) &data[0], data.size() * sizeof(double));
		depthout.close();
	}

	void Depth::updateStatics() {
		average_depth = 0;
		depth_var = 0;
		int count = 0;
		for (int i = 0; i < data.size(); i++) {
			if (data[i] >= 0) {
				average_depth += data[i];
				count++;
			}
		}
		if (count != 0) {
			average_depth /= static_cast<double>(count);
		}
		for (int i = 0; i < data.size(); i++) {
			if (data[i] >= 0)
				depth_var += std::sqrt((data[i] - average_depth) * (data[i] - average_depth));
		}
		depth_var /= static_cast<double>(count);

		vector<double> data_dummy = data;
		const size_t kth = data.size() / 2;
		nth_element(data_dummy.begin(), data_dummy.begin() + kth, data_dummy.end());
		median_depth = data_dummy[kth];
		statics_computed = true;
	}

	void Depth::fillholeAndSmooth() {
		vector<bool> mask(data.size());
		vector<double> data_copy(data.size());
		for(int i=0; i<data.size(); ++i) {
			data_copy[i] = data[i];
			mask[i] = (data[i] >= 0);
		}
		possionSmooth(data_copy, getWidth(), getHeight(), mask);
		data.swap(data_copy);
	}

	void Depth::fillhole() {
		//deep copy data to ensure thread safety
		vector<double> data_copy(data.size());
		vector<bool> mask(data.size());
		const double epsilon = 0.0001;
		for(size_t i=0; i<data.size(); ++i) {
			mask[i] = (data_copy[i] >= 0);
			if(data[i] != 0)
				data_copy[i] = 1.0 / data[i];
			else
				data_copy[i] = 1.0 / epsilon;
		}

		fillHoleLinear(data_copy, getWidth(), getHeight(), mask);
		const double max_depth = 5.0;
		const double min_depth = 0.1;
		for(size_t i = 0; i<data.size(); ++i){
			if(data_copy[i] <= 1.0 / max_depth)
				data[i] = max_depth;
			else if(data_copy[i] >= 1.0 / min_depth)
				data[i] = max_depth;
			else
				data[i] = 1.0 / data_copy[i];
		}
	}

	void fillHoleLinear(std::vector<double> &data, const int width, const int height, const std::vector<bool> &mask) {
		CHECK_EQ(data.size(), mask.size());
		CHECK_EQ(width * height, (int)data.size());
		int invalidnum = 0;

		//invalidcoord: size of invalidnum
		//invalidindx: size of depthnum

		vector<int> invalidcoord;
		vector<int> invalidindex(data.size());

		for (int i = 0; i < data.size(); i++) {
			if (!mask[i]) {
				invalidnum++;
				invalidcoord.push_back(i);
				invalidindex[i] = invalidnum - 1;
			}
		}
		//construct matrix A and B
		SparseMatrix<double> A(invalidnum, invalidnum);
		VectorXd B(invalidnum);
		for(int i=0; i<invalidnum; ++i)
			B[i] = 0.0;

		typedef Triplet<double> TripletD;
		vector<TripletD> triplets;
		triplets.reserve(invalidnum * 4);
		for (int i = 0; i < invalidnum; i++) {
			//(x,y) is the coordinate of invalid pixel
			int x = invalidcoord[i] % width;
			int y = invalidcoord[i] / width;
			int count = 0;
			if (y * width + x - 1 < data.size()) {
				count++;
				if (!mask[y * width + x - 1])
					triplets.push_back(TripletD(i, invalidindex[y * width + x - 1], -1));
				else
					B[i] += data[y * width + x - 1];
			}
			if (y * width + x + 1 < data.size()) {
				count++;
				if (!mask[y * width + x + 1])
					triplets.push_back(TripletD(i, invalidindex[y * width + x + 1], -1));
				else
					B[i] += data[y * width + x + 1];
			}
			if ((y - 1) * width + x < data.size()) {
				count++;
				if (!mask[(y - 1) * width + x])
					triplets.push_back(TripletD(i, invalidindex[(y - 1) * width + x], -1));
				else
					B[i] += data[(y - 1) * width + x];
			}
			if ((y + 1) * width + x < data.size()) {
				count++;
				if (!mask[(y + 1) * width + x])
					triplets.push_back(TripletD(i, invalidindex[(y + 1) * width + x], -1));
				else
					B[i] += data[(y + 1) * width + x];
			}
			triplets.push_back(TripletD(i, i, (double) count));
		}
		A.setFromTriplets(triplets.begin(), triplets.end());

		//Solve the linear problem
		SimplicialLDLT<SparseMatrix<double> > solver(A);
		VectorXd solution = solver.solve(B);
		for (int i = 0; i < invalidnum; i++)
			data[invalidcoord[i]] = solution[i];

	}

	void possionSmooth(std::vector<double> &data, const int width, const int height, const std::vector<bool> &mask, const double kDataCost) {
		CHECK_EQ(data.size(), mask.size());
		CHECK_EQ(width * height, data.size());

		//cout << "Possion smooth: Invalid depth num:" << invalidnum << endl;
		const int num_vars = (int)data.size();
		SparseMatrix<double> A(num_vars, num_vars);
		VectorXd b(num_vars);
		//data constraint
		vector<Triplet<double> > triplets;
		triplets.reserve(4 * num_vars);
		int index = 0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x, ++index) {
				double leftv = 0, rightv = 0;
				if(mask[index]) {
					leftv += kDataCost;
					rightv += kDataCost * data[index];
				}
				vector<int> neighbor_indexes;
				if (x != width - 1)
					neighbor_indexes.push_back(index + 1);
				if (x != 0)
					neighbor_indexes.push_back(index - 1);
				if (y != height - 1)
					neighbor_indexes.push_back(index + width);
				if (y != 0)
					neighbor_indexes.push_back(index - width);
				leftv += (double)neighbor_indexes.size();
				triplets.push_back(Triplet<double>(index, index, leftv));
				for (int i = 0; i < neighbor_indexes.size(); i++)
					triplets.push_back(Triplet<double>(index, neighbor_indexes[i], -1));
				b[index] = rightv;
			}
		}
		A.setFromTriplets(triplets.begin(), triplets.end());
		VectorXd x;
		SimplicialLDLT<SparseMatrix<double> > solver(A);
		x = solver.solve(b);
		for(int i=0; i<num_vars; ++i)
			data[i] = x[i];
	}


}










