//
//  3Dplane.cpp
//  DynamicOptimize
//
//  Created by Yan Hang on 10/16/14.
//  Copyright (c) 2014 Washington Universtiy. All rights reserved.
//

#include "plane3D.h"
#include <time.h>
#include <random>

using namespace std;
using namespace Eigen;

namespace dynamic_stereo {

	double Plane3D::epsilon = 1e-4;

	Plane3D::Plane3D(const Vector3d &p0, const Vector3d &p1, const Vector3d &p2) {
		Matrix3d A;
		Vector3d b(3);
		b[0] = -1; b[1] = -1; b[2] = -1;
		A.block<1,3>(0,0) = p0;
		A.block<1,3>(1,0) = p1;
		A.block<1,3>(2,0) = p2;
		const double Adet = A.determinant();
		if(Adet <= epsilon) {
			printf("Points colinear: (%.5f,%.5f,%.5f),(%.5f,%.5f,%.5f),(%.5f,%.5f,%.5f)\n", p0[0], p0[1], p0[2],
			       p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
			CHECK_NE(A.determinant(), 0);
		}
		normal = A.inverse() * b;
		const double norm = normal.norm();
		if(norm <= epsilon){
			printf("Bad condition: (%.5f,%.5f,%.5f),(%.5f,%.5f,%.5f),(%.5f,%.5f,%.5f)\n", p0[0], p0[1], p0[2],
			       p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
		}
		CHECK_GT(norm, 0) << endl << A;
		normal = normal / norm;
		offset = 1 / norm;
	}

	double Plane3D::getDistance(const Vector3d &p) const {
		double dis;
		double nn = getNormal().norm();
		CHECK_GE(nn, epsilon);
		dis = std::abs((p.dot(normal) + offset) / nn);
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

		bool planeFromPointsLeastSquare(const std::vector<Eigen::Vector3d> &pts, Plane3D &plane) {
			MatrixXd A(pts.size(), 3);
			VectorXd b(pts.size());
			for (auto i = 0; i < pts.size(); ++i) {
				A.block<1, 3>(i, 0) = pts[i];
				b[i] = -1;
			}
			Vector3d n = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
			const double epsilon = 1e-10;
			const double nn = n.norm();
			if(nn < epsilon)
				return false;
			n /= nn;
			Vector3d pt(0, 0, 0);
			if (n[0] != 0)
				pt[0] = (-1.0 / nn) / n[0];
			else if (n[1] != 0)
				pt[1] = (-1.0 / nn) / n[1];
			else
				pt[2] = (-1.0 / nn) / n[2];
			plane = Plane3D(pt, n);
			return true;
		}


		bool planeFromPointsRANSAC(const std::vector<Eigen::Vector3d> &pts, Plane3D &plane, const double dis_thres,
								   const int max_iter, bool verbose) {
			size_t max_inlier = 0;
			const double epsilon = 1e-5;
			int N = (int)pts.size();
			CHECK_GE(N, 3);
			if(verbose){
				printf("=========================\n");
				cout << "Solving plane RANSAC:" << endl;
//				for(int i=0; i<pts.size(); ++i)
//					cout << pts[i][0] << ' ' << pts[i][1] << ' ' << pts[i][2] << endl;

			}
			plane = Plane3D();
			std::default_random_engine generator;
			std::uniform_int_distribution<int> distribution(0, (int) pts.size() - 1);
			for(int iter=0; iter < max_iter; ++iter){
				int id1 = distribution(generator);
				int id2 = distribution(generator);
				int id3 = distribution(generator);

				if(id1 == id2 || id1 == id3 || id2 == id3) {
//					printf("Duplicate ids! %u, (%d,%d,%d)\n",seed, id1, id2, id3);
					iter--;
					continue;
				}
				if(verbose){
					printf("------------------\niter %d\n", iter);
					printf("id1:%d, id2:%d, id3:%d\n", id1, id2, id3);
					printf("%.5f,%.5f,%.5f\n", pts[id1][0], pts[id1][1], pts[id1][2]);
					printf("%.5f,%.5f,%.5f\n", pts[id2][0], pts[id2][1], pts[id2][2]);
					printf("%.5f,%.5f,%.5f\n", pts[id3][0], pts[id3][1], pts[id3][2]);
				}
				Eigen::Matrix3d A;
				A.block<1,3>(0,0) = pts[id1];
				A.block<1,3>(1,0) = pts[id2];
				A.block<1,3>(2,0) = pts[id3];
				if(A.determinant() < epsilon)
					continue;

				//check if three points are colinear
				if((pts[id1] - pts[id2]).cross(pts[id1]-pts[id3]).norm() < epsilon)
					continue;

				Plane3D curplane(pts[id1], pts[id2], pts[id3]);
//				printf("plane: (%.2f,%.2f,%.2f,%.2f)\n", curplane.getNormal()[0], curplane.getNormal()[1], curplane.getNormal()[2], curplane.getOffset());
				vector<Vector3d> inliers;
				for(int i=0; i<pts.size(); ++i){
					double dis = curplane.getDistance(pts[i]);
					if(dis < dis_thres) {
						//printf("inlier: (%.5f,%.5f,%.5f), dis:%.5f\n", pts[i][0], pts[i][1], pts[i][2], dis);
						inliers.push_back(pts[i]);
					}
				}
//				printf("inlier size: %lu, max_inlier:%lu\n", inliers.size(), max_inlier);
				if(inliers.size() > max_inlier) {
					if(!planeFromPointsLeastSquare(inliers, plane))
						continue;
					max_inlier = inliers.size();
					if(verbose){
						printf("new fitted plane: (%.2f,%.2f,%.2f,%.2f)\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset());
						printf("Max inlier: %lu\n", max_inlier);
					}
				}
//				printf("Current optimal plane: (%.2f,%.2f,%.2f,%.2f)\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset());
			}
			if(verbose){
				printf("Optimal plane: (%.5f,%.5f,%.5f,%.5f), inlier count: %lu, total count: %lu\n", plane.getNormal()[0], plane.getNormal()[1], plane.getNormal()[2], plane.getOffset(),
				       max_inlier, pts.size());
			}


			if(plane.getNormal().norm() < epsilon)
				return false;
			return true;
		}

	}//namespace plane_util
}//namespace dynamic_stereo
