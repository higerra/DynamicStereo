//
// Created by Yan Hang on 1/20/16.
// Based on c++ code of Daniel Scharstein
// http://vision.middlebury.edu/flow/data/
//

#ifndef RENDERPROJECT_COLORWHEEL_H
#define RENDERPROJECT_COLORWHEEL_H
#include <Eigen/Eigen>
#include <memory>
#include <vector>

namespace dynamic_stereo {
	class ColorWheel {
	public:
		static std::shared_ptr<ColorWheel> instance() {
			return ins;
		}

		Eigen::Vector3d computeColor(const Eigen::Vector2d &fv);

		ColorWheel(const ColorWheel &w) = delete;

		ColorWheel &operator==(const ColorWheel &w) = delete;

	private:
		ColorWheel() : MAXCOLS(60) {
			makewheel();
		};

		static std::shared_ptr<ColorWheel> ins;

		inline void setcols(int r, int g, int b, int k) {
			colorwheel[k][0] = r;
			colorwheel[k][1] = g;
			colorwheel[k][2] = b;
		}

		void makewheel();

		std::vector<std::vector<int> > colorwheel;
		const int MAXCOLS;
		int ncols;
	};

}
#endif //RENDERPROJECT_COLORWHEEL_H
