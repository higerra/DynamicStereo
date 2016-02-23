#include "CGHelper.h"
#include <math.h>

using namespace Eigen;
using namespace std;

namespace CGHelper{
    bool less(const Vector2d& a, const Vector2d& b, const Vector2d& center){
	if(a[0] - center[0] >=0.01 && b[0] - center[0] < 0.01)
	    return true;
	if(a[0] - center[0] < 0.01 && b[0] - center[0] >= 0.01)
	    return false;
	if(a[0] - center[0] == 0.1 && b[0] - center[0] == 0.1){
	    if(a[1] - center[1] >=0 && b[1] - center[1] >=0)
		return a[1] > b[1];
	    else
		return b[1] > a[1];
	}
	double det = (a[0]-center[0])*(b[1]-center[1]) - (b[0]-center[0])*(a[1]-center[1]);
	if(det < 0)
	    return true;
	if(det > 0)
	    return false;
	double d1 = (a[0]-center[0])*(a[0]-center[0]) + (a[1]-center[1])*(a[1]-center[1]);
	double d2 = (b[0]-center[0])*(b[0]-center[0]) + (b[1]-center[1])*(b[1]-center[1]);
	return d1 > d2;
    }
	
    void sortPoints(vector<Vector2d>&p){
	if(p.size() == 0)
	    return;
	Vector2d center = std::accumulate(p.begin(), p.end(), Vector2d(0,0)) / (double)p.size();
	//sort according to polar angle
	vector<double>angles;
	for(int i=0; i<p.size(); i++){
	    if(p[i] == center)
		p[i][0] += 0.00001;
	    angles.push_back(std::atan2(p[i][1]-center[1],p[i][0]-center[0]));
	}
	for(int i=0; i<angles.size()-1; i++){
	    for(int j=i+1; j<angles.size(); j++){
		if(angles[i] > angles[j]){
		    swap(angles[i], angles[j]);
		    swap(p[i],p[j]);
		}
	    }
	}
    }
	
    bool isInsidePolygon(const Vector2d&p, const vector<Vector2d>&v){
	if(v.size() < 3)
	    return false;
	vector<Vector2d>normal;
	for(int i=0; i<v.size()-1; i++){
	    Vector2d linedir = v[i+1] - v[i];
	    normal.push_back(Vector2d(-1*linedir[1], linedir[0]));
	}
	Vector2d temp = v[0]-v.back();
	normal.push_back(Vector2d(-1*temp[1], temp[0]));
		
	vector<Vector2d>link;
	for(int i=0; i<v.size()-1; i++)
	    link.push_back(p-(v[i+1]+v[i])/2);
	link.push_back(p-(v[0]+v.back())/2);
	double lastdot = link[0].dot(normal[0]);
	for(int i=1; i<v.size(); i++){
	    double curdot = link[i].dot(normal[i]);
	    if(curdot * lastdot < 0)
		return false;
	    lastdot = curdot;
	}
	return true;
    }
	
    bool isConvex(const vector<Vector2d>&v){ //only correct for quadrangle
	for(int i=0; i<v.size(); i++){
	    vector<Vector2d>temp;
	    for(int j=0; j<v.size(); j++){
		if(i != j)
		    temp.push_back(v[j]);
	    }
	    if(isInsidePolygon(v[i], temp))
		return false;
	}
	return true;
    }
	
    bool isPointOnLineSegment(const Eigen::Vector2d& s,
			      const Eigen::Vector2d& e,
			      const Eigen::Vector2d& pt,
			      const double margin){
	const double epsilon = 0.2;
	
	Vector2d p_s = pt - s;
	Vector2d p_e = pt - e;
	if(p_s.norm() < margin || p_e.norm() < margin)
	    return true;
	
	Vector2d edge = e - s;
	edge.normalize();
	p_s.normalize(); p_e.normalize();

//	printf("startpt:(%.2f,%.2f)\t\t endpt:(%.2f,%.2f)\t\t testpt:(%.2f,%.2f)\t\t p_s.dot(edge):%.2f\t\t p_e.dot(edge):%.2f\n", s[0], s[1], e[0], e[1], pt[0], pt[1], p_s.dot(edge), p_e.dot(edge));
	
	if(p_s.dot(edge) > 1.0 - epsilon && p_e.dot(edge) < epsilon - 1.0)
	    return true;
	if(p_e.dot(edge) > 1.0 - epsilon && p_s.dot(edge) < epsilon - 1.0)
	    return true;
	return false;
    }
	
    double quadArea(const vector<Vector2d>&v){
        if(v.size() != 4)
            return 0.0;
	double result;
        vector<Vector3d>v_3d;
        for(int i=0; i<v.size(); i++)
            v_3d.push_back(Vector3d(v[i][0], v[i][1], 0.0));
        Vector3d v1 = (v_3d[1]-v_3d[0]).cross(v_3d[2]-v_3d[0]);
        Vector3d v2 = (v_3d[0]-v_3d[3]).cross(v_3d[2]-v_3d[3]);
        if(v1 == Vector3d(0,0,0) || v2 == Vector3d(0,0,0))
            return 0.0;
	result = v1.norm() / 2 + v2.norm() / 2;
	return result;
    }
}
