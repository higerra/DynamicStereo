//
//  3Dplane.cpp
//  DynamicOptimize
//
//  Created by Yan Hang on 10/16/14.
//  Copyright (c) 2014 Washington Universtiy. All rights reserved.
//

#include "plane3D.h"
#include <time.h>

using namespace std;
using namespace Eigen;

namespace dynamic_stereo {
	Plane3D::Plane3D(const Vector3d &p0, const Vector3d &p1, const Vector3d &p2) {
		Vector3d v1 = p1 - p0;
		Vector3d v2 = p2 - p0;

		//check if collinear points
		double cosinangle = v1.dot(v2) / (v1.norm() * v2.norm());
		if (cosinangle > 0.995 || cosinangle < -0.995) {
			normal = Vector3d(0, 0, 0);
			offset = 0;
			return;
		}
		normal = v1.cross(v2);
		normal = normal / normal.norm();
		offset = -1 * (normal.dot(p0));
	}

	double Plane3D::getDistance(const Vector3d &p) const {
		double dis;
		dis = p.dot(normal) + offset;
		return dis;
	}

	Vector3d Plane3D::projectFromeWorldtoPlane(const Vector3d &pt) const {
		double dis = getDistance(pt);
		Vector3d newpt = pt - normal * dis;
		return newpt;
	}

	namespace plane_util {
		void planeIntersection(const Plane3D &plane1,
							   const Plane3D &plane2,
							   Vector3d &normal,
							   Vector3d &pt) {
			Vector3d n1 = plane1.getNormal();
			Vector3d n2 = plane2.getNormal();
			normal = n1.cross(n2);
			CHECK_GT(normal.norm(), 0);

			normal.normalize();
			Matrix2d A;
			Vector2d p(-1 * plane1.getOffset(), -1 * plane2.getOffset());
			Vector2d xy;
			if (normal[2] != 0) {
				A << n1[0], n1[1], n2[0], n2[1];
				xy = A.inverse() * p;
				pt[0] = xy[0];
				pt[1] = xy[1];
				pt[2] = 0.0;
				return;
			} else if (normal[0] != 0) {
				A << n1[1], n1[2], n2[1], n2[2];
				xy = A.inverse() * p;
				pt[0] = 0.0;
				pt[1] = xy[0];
				pt[2] = xy[1];
				return;
			} else if (normal[1] != 0) {
				A << n1[0], n1[2], n2[0], n2[2];
				xy = A.inverse() * p;
				pt[0] = xy[0];
				pt[1] = 0.0;
				pt[2] = xy[1];
				return;
			}
		}

		void planeFromPointsLeastSquare(const std::vector<Eigen::Vector3d> &pts, Plane3D &plane) {
			MatrixXd A(pts.size(), 3);
			VectorXd b(pts.size());
			for (auto i = 0; i < pts.size(); ++i) {
				A.block<1, 3>(i, 0) = pts[i];
				b[i] = -1;
			}
			Vector3d n = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
			CHECK_GT(n.norm(), 0);
			Vector3d pt(0, 0, 0);
			if (n[0] != 0)
				pt[0] = -1.0 / n[0];
			else if (n[1] != 0)
				pt[1] = -1.0 / n[1];
			else
				pt[2] = -1.0 / n[2];
			plane = Plane3D(pt, n);
		}


		void planeFromPointsRANSAC(const std::vector<Eigen::Vector3d> &pts, Plane3D &plane, const double dis_thres,
								   const int max_iter) {
			size_t max_inlier = 0;
			int N = (int)pts.size();
			CHECK_GE(N, 3);
//			cout << "Solving plane RANSAC:" << endl;
//			for(int i=0; i<pts.size(); ++i)
//				cout << pts[i][0] << ' ' << pts[i][1] << ' ' << pts[i][2] << endl;
			if(N == 3){
				plane = Plane3D(pts[0], pts[1], pts[2]);
				return;
			}
			unsigned int seed = 0;
			for(int iter=0; iter < max_iter; ++iter){
				srand(seed++);
				int id1 = rand() % N;
				srand(seed++);
				int id2 = rand() % N;
				srand(seed++);
				int id3 = rand() % N;
				if(id1 == id2 || id1 == id3 || id2 == id3) {
					iter--;
					continue;
				}
//				printf("======================\niter %d\n", iter);
//				printf("id1:%d, id2:%d, id3:%d\n", id1, id2, id3);
				Plane3D curplane(pts[id1], pts[id2], pts[id3]);
//				printf("plane: (%.2f,%.2f,%.2f,%.2f)\n", curplane.getNormal()[0], curplane.getNormal()[1], curplane.getNormal()[2], curplane.getOffset());
				vector<Vector3d> inliers;
				for(int i=0; i<pts.size(); ++i){
					double dis = curplane.getDistance(pts[i]);
					if(dis < dis_thres)
						inliers.push_back(pts[i]);
				}
//				printf("inlier size: %lu, max_inlier:%lu\n", inliers.size(), max_inlier);
				if(inliers.size() > max_inlier) {
					planeFromPointsLeastSquare(inliers, plane);
					max_inlier = inliers.size();
//					printf("new fitted plane: (%.2f,%.2f,%.2f,%.2f)\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset());
//					printf("Max inlier: %lu\n", max_inlier);
				}
//				printf("Current optimal plane: (%.2f,%.2f,%.2f,%.2f)\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset());
			}
//			printf("Optimal plane: (%.2f,%.2f,%.2f,%.2f)\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset());
		}

	}//namespace plane_util
}//namespace dynamic_stereo
