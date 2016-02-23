#ifndef CGHELPER_H
#define CGHELPER_H

#include <Eigen/Eigen>
#include <iostream>
#include <numeric>
#include <vector>

namespace CGHelper{
    void sortPoints(std::vector<Eigen::Vector2d>&p);
    bool isPointOnLineSegment(const Eigen::Vector2d& s,
			      const Eigen::Vector2d& e,
			      const Eigen::Vector2d& pt,
			      const double margin = 20.0);
    bool isInsidePolygon(const Eigen::Vector2d& p, const std::vector<Eigen::Vector2d>& v);
    bool isConvex(const std::vector<Eigen::Vector2d>&v);
    double quadArea(const std::vector<Eigen::Vector2d>& v);
}


#endif
