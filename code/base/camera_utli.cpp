#include "camera_utli.h"
#include <iostream>
using namespace std;
using namespace Eigen;

namespace dynamic_rendering{

    namespace camera_utility{
	Matrix4d inverseExtrinsic(const Matrix4d& mat){
	    Vector3d tvec(mat(0,3), mat(1,3), mat(2,3));
	    Matrix3d rot = mat.block<3,3>(0,0);
	    rot.transposeInPlace();
	    tvec = -1 * rot * tvec;
	    Matrix4d inv;
	    inv << rot(0,0), rot(0,1), rot(0,2), tvec[0],
		rot(1,0), rot(1,1), rot(1,2), tvec[1],
		rot(2,0), rot(2,1), rot(2,2), tvec[2],
		0,0,0,1;
	    return inv;
	}

	struct triangulationFunc{
	    triangulationFunc(const double x_, const double y_, Matrix4d projection_):measured_x(x_), measured_y(y_), p(projection_){}

	    template<typename T>
	    bool operator()(const T* const p1, const T* const p2, const T* const p3, const T* const p4, T* residual) const{

		T repo_x = p(0,0) * p1[0] + p(0,1) * p2[0] + p(0,2) * p3[0] + p(0,3) * p4[0];
		T repo_y = p(1,0) * p1[0] + p(1,1) * p2[0] + p(1,2) * p3[0] + p(1,3) * p4[0];
		T repo_w = p(2,0) * p1[0] + p(2,1) * p2[0] + p(2,2) * p3[0] + p(2,3) * p4[0];


		if(repo_w != (T)0){
		    repo_x /= repo_w;
		    repo_y /= repo_w;
		}else{
		    residual[0] = (T)numeric_limits<double>::max();
		    residual[1] = (T)numeric_limits<double>::max();
		    return true;
		}

		residual[0] = repo_x - (T)measured_x;
		residual[1] = repo_y - (T)measured_y;
		return true;
	    }

	    static ceres::CostFunction* create(const double x_, const double y_, Matrix4d projection_){
		return (new ceres::AutoDiffCostFunction<triangulationFunc,2,1,1,1,1>(new triangulationFunc(x_, y_, projection_)));
	    }

	private:
	    double measured_x;
	    double measured_y;
	    Matrix4d p;
	};

	void triangulation(const vector<Eigen::Vector2d>& pt,
			   const vector<Camera>& cam,
			   Vector3d& spacePt,
			   double& residual,
			   const bool min_repojection_error){
	    //Get initial point by linear triangulation
	    MatrixXd A(pt.size() * 2, 4);
	    for(int i=0; i<pt.size(); ++i){
		const Matrix4d& curP = cam[i].getProjection();
		A.block<1,4>(2*i,0) = pt[i][0]*curP.block<1,4>(2,0) - curP.block<1,4>(0,0);
		A.block<1,4>(2*i+1, 0) = pt[i][1]*curP.block<1,4>(2,0) - curP.block<1,4>(1,0);
	    }
	    JacobiSVD<MatrixXd>svd(A, ComputeThinV);
	    vector<double>spacePt_homo(4);
	    // Matrix4d U = svd.matrixU();
	    Matrix4d V = svd.matrixV();

	    for(int i=0; i<4; i++) spacePt_homo[i] = V(i,3);

	    if(min_repojection_error) {
		//minimize reprojection error
		ceres::Problem problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
		options.max_num_iterations = 1000;

		for (int i = 0; i < pt.size(); i++) {
		    ceres::CostFunction *cost_function = triangulationFunc::create(pt[i][0], pt[i][1],
										   cam[i].getProjection());
		    problem.AddResidualBlock(cost_function, NULL, &spacePt_homo[0], &spacePt_homo[1], &spacePt_homo[2],
					     &spacePt_homo[3]);
		}
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		residual = summary.final_cost;
	    }else
		residual = -1;

	    if(spacePt_homo[3] != 0){
		spacePt[0] = spacePt_homo[0] / spacePt_homo[3];
		spacePt[1] = spacePt_homo[1] / spacePt_homo[3];
		spacePt[2] = spacePt_homo[2] / spacePt_homo[3];
	    }else{
		spacePt[0] = 0;
		spacePt[1] = 0;
		spacePt[2] = 0;
	    }
	}
    } //namespace camera_utility
} //namesapce dynamic_rendering
