#ifndef QUAD_UTIL_H
#define QUAD_UTIL_H

#include "quad.h"
#include <opencv2/opencv.hpp>

namespace dynamic_rendering{
    class Frame;
    class Depth;
    namespace quad_util{
	
	Eigen::Matrix3d covariance3d(std::vector<Eigen::Vector3d> &m, const double ratio = 1.0);
//lines have to be counterclock
	double getLinesColorVar(const Frame& frame,
				const std::vector<Eigen::Vector2d>& spt,
				const std::vector<Eigen::Vector2d>& ept,
				double offset = 3, const double ratio = 1.0,
				bool inside = true);

	double getQuadColorVar(const Frame& frame,
			       const Quad& quad,
			       const double offset = 3, const double ratio = 1.0,
			       bool inside = true);

	double getQuadColorDiff(const Quad& q1, const Quad& q2, const Frame& frame1, const Frame& frame2, const double offset=3.0, bool inside = true);

	double getQuadShapeDiff(const Quad& q1, const Quad& q2);

	double outsideRatio(const Frame& frame,
			    const Quad& quad);


	void drawQuad(const Quad& quad, cv::Mat& image, bool is_color = false, int thickness=2.0);

	void drawSingleLine(const KeyLine& line,
			    const int lid,
			    cv::Mat& image,
			    const cv::Scalar& color,
			    int thickness = 1);

	void drawLineGroup(const std::vector<std::vector<KeyLine> >& line_group,
			   cv::Mat& image,
			   int thickness = 2.0);
	
	void getQuadDepth(const Quad& q,
			  const Frame& frame,
			  std::vector<double>& dv);

    }
}

#endif
