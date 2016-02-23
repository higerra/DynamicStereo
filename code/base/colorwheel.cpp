//
// Created by Yan Hang on 1/20/16.
//

#include "colorwheel.h"
using namespace std;
using namespace Eigen;

namespace dynamic_stereo {
	shared_ptr<ColorWheel> ColorWheel::ins(new ColorWheel());

	void ColorWheel::makewheel() {
		colorwheel.resize(MAXCOLS);
		for (auto &v: colorwheel)
			v.resize(3);
		int RY = 15;
		int YG = 6;
		int GC = 4;
		int CB = 11;
		int BM = 13;
		int MR = 6;
		ncols = RY + YG + GC + CB + BM + MR;
		int i;
		int k = 0;
		for (i = 0; i < RY; i++) setcols(255, 255 * i / RY, 0, k++);
		for (i = 0; i < YG; i++) setcols(255 - 255 * i / YG, 255, 0, k++);
		for (i = 0; i < GC; i++) setcols(0, 255, 255 * i / GC, k++);
		for (i = 0; i < CB; i++) setcols(0, 255 - 255 * i / CB, 255, k++);
		for (i = 0; i < BM; i++) setcols(255 * i / BM, 0, 255, k++);
		for (i = 0; i < MR; i++) setcols(255, 0, 255 - 255 * i / MR, k++);
	}

	Eigen::Vector3d ColorWheel::computeColor(const Eigen::Vector2d &fv) {
		const double fx = fv[0];
		const double fy = fv[1];
		if (colorwheel.empty())
			makewheel();
		double rad = sqrt(fx * fx + fy * fy);
		double a = atan2(-fy, -fx) / M_PI;
		double fk = (a + 1.0) / 2.0 * (ncols - 1);
		int k0 = (int) fk;
		int k1 = (k0 + 1) % ncols;
		double f = fk - k0;
		//f = 0; // uncomment to see original color wheel
		Vector3d res;
		for (int b = 0; b < 3; b++) {
			double col0 = colorwheel[k0][b] / 255.0;
			double col1 = colorwheel[k1][b] / 255.0;
			double col = (1 - f) * col0 + f * col1;
			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= .75; // out of range
			res[2 - b] = 255.0 * col;
		}
		return res;
	}

}//namespace