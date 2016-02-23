#ifndef QUAD_H
#define QUAD_H
#include <iostream>
#include <vector>
#include <Eigen/Eigen>

namespace dynamic_rendering{
struct Quad{
Quad():frameid(-1),rank(0), confidence(0), colorvar(0), score(0), aux(0), counter(0){
    cornerpt.resize(4), lineind.resize(4), cluster.resize(4);
}
    std::vector<Eigen::Vector2d> cornerpt;
    int frameid;
    int rank;
    double confidence;
    double colorvar;
    double score;
    double aux;
    int counter;
    std::vector<int> lineind;
    std::vector<int> cluster;
    static bool compareQuad_score(const Quad& lhs, const Quad& rhs){
	return lhs.score > rhs.score;
    }

    static bool compareQuad_colorvar(const Quad& lhs, const Quad& rhs){
	return lhs.colorvar < rhs.colorvar;
    }
    inline double getDiagonal() const{
	return ((cornerpt[0]-cornerpt[2]).norm() + (cornerpt[1]-cornerpt[3]).norm())/2;
    }
    void toCardinal();
};


inline void dehomoLine(Eigen::Vector3d& v){
    double normal_factor = std::sqrt(v[0]*v[0] + v[1]*v[1]);
    if(normal_factor > 0){
	v[0] /= normal_factor;
	v[1] /= normal_factor;
	v[2] /= normal_factor;
    }
}

inline void dehomoPoint(Eigen::Vector3d& pt){
    const double epsilon = 0.00001;
    if(std::abs(pt[2]) > epsilon){
	pt[0] /= pt[2];
	pt[1] /= pt[2];
	pt[2] = 1.0;
    }
}

inline Eigen::Vector2d getDehomoPoint(Eigen::Vector3d pt){
    dehomoPoint(pt);
    if(pt[2] > 0.99)
	return Eigen::Vector2d(pt[0], pt[1]);
    else
	return Eigen::Vector2d(0,0);
}


struct KeyLine{
    Eigen::Vector2d startPoint;
    Eigen::Vector2d endPoint;
    double lineLength;
    
    KeyLine():lineLength(0.0){}
    KeyLine(Eigen::Vector2d start_, Eigen::Vector2d end_):startPoint(start_), endPoint(end_){
	if(startPoint != endPoint)
	    lineLength = (endPoint - startPoint).norm();
	else{
	    std::cerr << "Start point cannot be the same with end point!"<<std::endl;
	    exit(-1);
	}
    }
    inline Eigen::Vector2d getLineDir() const{
	if(lineLength == 0.0)
	    return Eigen::Vector2d(0,0);
	Eigen::Vector2d linedir = startPoint - endPoint;
	linedir.normalize();
	return linedir;
    }

    inline double getLength(){
	lineLength = (endPoint - startPoint).norm();
	return lineLength;
    }
    
    Eigen::Vector3d getHomo()const{
	if(lineLength == 0.0)
	    return Eigen::Vector3d(0,0,0);
	Eigen::Vector3d s(startPoint[0], startPoint[1], 1.0);
	Eigen::Vector3d e(endPoint[0], endPoint[1], 1.0);
	Eigen::Vector3d line_homo = s.cross(e);
	dehomoLine(line_homo);
	return line_homo;
    }
    
    double distanceToPoint(const Eigen::Vector2d&pt)const {
	Eigen::Vector2d v12 = endPoint - startPoint;
	v12.normalize();
	Eigen::Vector2d v10 = pt - startPoint;
	Eigen::Vector2d p3 = startPoint + v12 * (v10.dot(v12));
	Eigen::Vector2d v30 = pt - p3;
	return v30.norm();
    }
};

std::ostream& operator << (std::ostream& ostr, const Quad& quad);
std::istream& operator >> (std::istream& istr, Quad& quad);

double quadDistance(const Quad& q1, const Quad& q2);
double quadMaxDistance(const Quad& q1, const Quad& q2);

//linear when v is small, quadratic when v is large. Opposite to Huber loss
inline double inverseHuberLoss(const double v, const double pivot, const double k = 1.0){
    if(pivot <= 0)
	return v * v; 
    if(v <= pivot)
	return k * v;
    double p =  k * pivot;
    double b = k - 2 * pivot;
    double c = p - pivot * pivot - b * pivot;
    return v * v + b * v + c;
}

} //namespace dynamic_rendering
#endif










