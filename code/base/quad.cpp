#include "quad.h"
#include <fstream>
using namespace std;
using namespace Eigen;

namespace dynamic_rendering{
	std::ostream& operator << (std::ostream& ostr, const Quad& quad){
		ostr << quad.rank<<' '<<quad.frameid<<' ';
		for(int i=0; i<quad.cornerpt.size(); i++){
			const Vector2d& curpt = quad.cornerpt[i];
			ostr << curpt[0] <<' '<<curpt[1]<<' ';
		}
		ostr << quad.cluster[0] <<' '<<quad.cluster[1] <<' '<<quad.cluster[2]<<' '<<quad.cluster[3];
		ostr << '\t'<<quad.confidence <<'\t'<<quad.colorvar<<'\t'<<quad.score;
		return ostr;
	}

	std::istream& operator >> (std::istream& istr, Quad& quad){
		istr >> quad.rank>>quad.frameid;
		quad.cornerpt.resize(4);
		for(int i=0; i<4; i++){
			Vector2d& curpt = quad.cornerpt[i];
			istr >> curpt[0] >> curpt[1];
		}
		istr >> quad.cluster[0] >> quad.cluster[1] >> quad.cluster[2] >> quad.cluster[3];
		istr >> quad.confidence >> quad.colorvar >> quad.score;
		return istr;
	}

	double quadDistance(const Quad& q1, const Quad& q2){
		double res = 0.0;
		for(int j=0; j<4; j++){
			res += (q1.cornerpt[j] - q2.cornerpt[j]).norm();
		}
		return res;
	}


	double quadMaxDistance(const Quad& q1, const Quad& q2){
		double res = 0.0;
		for(int j=0; j<4; j++){
			res = std::max(res, (q1.cornerpt[j] - q2.cornerpt[j]).norm());
		}
		return res;
	}

}//namespace dynamic_rendering
