//
//  3Dplane.cpp
//  DynamicOptimize
//
//  Created by Yan Hang on 10/16/14.
//  Copyright (c) 2014 Washington Universtiy. All rights reserved.
//

#include "plane3D.h"

using namespace std;
using namespace Eigen;

namespace dynamic_rendering{
	Plane3D::Plane3D(const Vector3d& p0, const Vector3d& p1,const Vector3d& p2){
		Vector3d v1 = p1 - p0;
		Vector3d v2 = p2 - p0;

		//check if collinear points
		double cosinangle = v1.dot(v2)/(v1.norm()*v2.norm());
		if(cosinangle > 0.95 || cosinangle < -0.95){
			normal = Vector3d(0,0,0);
			offset = 0;
			return;
		}
		normal = v1.cross(v2);
		normal = normal/normal.norm();
		offset = -1*(normal.dot(p0));
	}

	void Plane3D::buildCoordinate(const Vector3d& xaxis_, double planesize,int width,int height){
		xaxis = xaxis_;
		xaxis = (Matrix3d::Identity()-normal*normal.transpose())*xaxis;
		yaxis = normal.cross(xaxis);
		xaxis = xaxis/xaxis.norm()*planesize/width;
		yaxis = yaxis/yaxis.norm()*planesize/width;
		planewidth = width;
		planeheight = height;
	}

	double Plane3D::getDistance(const Vector3d& p) const{
		double dis;
		dis = p.dot(normal)+offset;
		return dis;
	}

	Vector3d Plane3D::getWorldCoordinate(const Vector2i& planept) const{
		Vector3d result = planecenter+xaxis*(planept[0]-planewidth/2)+yaxis*(planept[1]-planeheight/2);
		return result;
	}

	Vector3d Plane3D::projectFromeWorldtoPlane(const Vector3d& pt) const{
		double dis = getDistance(pt);
		Vector3d newpt = pt - normal*dis;
		return newpt;
	}

	Vector2i Plane3D::getPlaneCoordinate(const Vector3d& pt) const{
		Vector3d planept = projectFromeWorldtoPlane(pt);
		Vector3d planevector = planept - planecenter;
		double xlength = xaxis.dot(planevector)/xaxis.norm();
		double ylength = yaxis.dot(planevector)/yaxis.norm();
		Vector2i planecoord;
		planecoord[0] = static_cast<int>(xlength / xaxis.norm() + planewidth/2);
		planecoord[1] = static_cast<int>(ylength / yaxis.norm() + planeheight/2);
		return planecoord;
	}

	void Plane3D::rotation(double angle){
		Matrix3d ncross;
		ncross<<0,-1*normal[2],normal[1],normal[2],0,-1*normal[0],-1*normal[1],normal[0],0;
		Matrix3d rotationmatrix = Matrix3d::Identity() + sinf(angle)*ncross + (1-cosf(angle))*ncross*ncross;
		xaxis = rotationmatrix * xaxis;
		yaxis = rotationmatrix * yaxis;
	}

	void planeIntersection(const Plane3D& plane1,
						   const Plane3D& plane2,
						   Vector3d& normal,
						   Vector3d& pt){
		Vector3d n1 = plane1.getNormal();
		Vector3d n2 = plane2.getNormal();
		normal = n1.cross(n2);
		assert(normal.norm() != 0);

		normal.normalize();
		Matrix2d A;
		Vector2d p(-1*plane1.getOffset(), -1*plane2.getOffset());
		Vector2d xy;
		if(normal[2] != 0){
			A << n1[0],n1[1],n2[0],n2[1];
			xy = A.inverse() * p;
			pt[0] = xy[0]; pt[1] = xy[1]; pt[2] = 0.0;
			return;
		}else if(normal[0] != 0){
			A << n1[1],n1[2],n2[1],n2[2];
			xy = A.inverse() * p;
			pt[0] = 0.0; pt[1] = xy[0]; pt[2] = xy[1];
			return;
		}else if(normal[1] != 0){
			A << n1[0],n1[2],n2[0],n2[2];
			xy = A.inverse() * p;
			pt[0] = xy[0]; pt[1] = 0.0; pt[2] = xy[1];
			return;
		}
	}
}
