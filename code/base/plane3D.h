#ifndef PLANE_3D_H
#define PLANE_3D_H

#include <iostream>
#include <Eigen/Eigen>


namespace dynamic_rendering{
	class Plane3D{

	private:
		Eigen::Vector3d normal;
		double offset;
		Eigen::Vector3d xaxis;
		Eigen::Vector3d yaxis;
		Eigen::Vector3d planecenter;
		Eigen::Vector3d planeorigin;
		int planewidth;
		int planeheight;
	public:
		Plane3D(){}
		Plane3D(const Eigen::Vector3d&,const Eigen::Vector3d&, const Eigen::Vector3d&);
		Plane3D(const Eigen::Vector3d& pt_, const Eigen::Vector3d& normal_): normal(normal_){
			offset = -1 * pt_.dot(normal);
		}
		//get and set
		inline const Eigen::Vector3d& getNormal()const {return normal;}
		inline const Eigen::Vector3d& getXaxis()const {return xaxis;}
		inline const Eigen::Vector3d& getYaxis()const {return yaxis;}
		inline const Eigen::Vector3d& getPlanecenter()const {return planecenter;}
		inline double getOffset()const {return offset;}
		inline int getWidth()const {return planewidth;}
		inline int getHeight()const {return planeheight;}
		
		void setWidth(int width){planewidth = width;}
		void setHeight(int height){planeheight = height;}
		void setNormal(const Eigen::Vector3d& n){normal = n;}
		void setOffset(double o){offset = 0;}
		void setXaxis(const Eigen::Vector3d& x){xaxis = x;}
		void setYaxis(const Eigen::Vector3d& y){yaxis = y;}
		void setPlanecenter(const Eigen::Vector3d& center){planecenter = center;}


		//plane operation
		void rotation(double angle);
		double getDistance(const Eigen::Vector3d& pt) const;
		Eigen::Vector3d getWorldCoordinate(const Eigen::Vector2i& pt)const;
		Eigen::Vector2i getPlaneCoordinate(const Eigen::Vector3d& pt)const;
		Eigen::Vector3d projectFromeWorldtoPlane(const Eigen::Vector3d& planept) const;
		void buildCoordinate(const Eigen::Vector3d& xaxis_, double planesize,int width,int height);
	};

	void planeIntersection(const Plane3D& plane1,
						   const Plane3D& plane2,
						   Eigen::Vector3d& normal,
						   Eigen::Vector3d& pt);

}
#endif










