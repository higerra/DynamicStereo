#ifndef PLANE_3D_H
#define PLANE_3D_H

#include <iostream>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace dynamic_stereo {
	class Plane3D {

	public:
		Plane3D():normal(0,0,0){}

		Plane3D(const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &);

		Plane3D(const Eigen::Vector3d &pt_, const Eigen::Vector3d &normal_) : normal(normal_) {
			CHECK_LT(normal_.norm() - 1.0, epsilon);
			offset = -1 * pt_.dot(normal);
		}

		inline void normalize(){
			double norm = normal.norm();
			CHECK_GT(norm, 0);
			normal = normal / norm;
			offset = offset / norm;
		}

		//get and set
		inline const Eigen::Vector3d &getNormal() const { return normal; }

		inline double getOffset() const { return offset; }

		void setNormal(const Eigen::Vector3d &n) { normal = n; }

		void setOffset(double o) { offset = o; }

		double getDistance(const Eigen::Vector3d &pt) const;
		Eigen::Vector3d projectFromeWorldtoPlane(const Eigen::Vector3d &planept) const;

	private:
		Eigen::Vector3d normal;
		double offset;
		static double epsilon;
	};

	namespace plane_util {
		void planeIntersection(const Plane3D &plane1,
							   const Plane3D &plane2,
							   Eigen::Vector3d &normal,
							   Eigen::Vector3d &pt);
		bool PlaneFromPointsLeastSquare(const std::vector<Eigen::Vector3d>& pts, Plane3D& plane);
		bool planeFromPointsRANSAC(const std::vector<Eigen::Vector3d>& pts, Plane3D& plane, const double dis_thres, const int max_iter = 500, bool verbose = false);
	}

}
#endif