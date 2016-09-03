//
// Created by yanhang on 9/2/16.
//

#include <limits>
#include "RPCA.h"
#include "../base/utility.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{

    static MatrixXd posMat(const MatrixXd& m){
        MatrixXd res = m;
#pragma omp parallel for
        for(auto y=0; y<m.rows(); ++y){
            for(auto x=0; x<m.cols(); ++x){
                if(m(y,x) < 0)
                    res(y,x) = 0;
            }
        }
    }

    static MatrixXd signMat(const MatrixXd& m){
        MatrixXd res = MatrixXd::Zero(m.rows(), m.cols());
#pragma omp parallel for
        for(auto y=0; y<m.rows(); ++y){
            for(auto x=0; x<m.cols(); ++x){
                if(m(y,x) > 0)
                    res(y,x) = 1;
                else if(m(y,x) < 0)
                    res(y,x) = -1;
            }
        }
    }

    void solveRPCA(const Eigen::MatrixXd& D, Eigen::MatrixXd& A_hat, Eigen::MatrixXd& E_hat, int& numIter,
                   const RPCAOption& option){
        const int maxLineSearchIter = 200;
        const int m = D.rows();
        const int n = D.cols();

        //initialization
        double t_k = 1, t_km1 = 1, tau_0 = 2;
        MatrixXd X_km1_A = MatrixXd::Zero(m, n);
        MatrixXd X_km1_E = MatrixXd::Zero(m, n);
        MatrixXd X_k_A = MatrixXd::Zero(m, n);
        MatrixXd X_k_E  = MatrixXd::Zero(m, n);

        double mu_k = option.mu, mu_0, mu_bar;
        if(option.continuationFlag){
            if(option.lineSearhFlag)
                mu_0 = option.eta * math_util::matrix2Norm<MatrixXd>(D);
            else
                mu_0 = 0;
             mu_k = 0.99 * mu_0;
            mu_bar = 1e-9 * mu_0;
        }

        double tau_k = tau_0;
        bool converged = false;
        numIter = 0;

        double stoppingCriterionOld = numeric_limits<double>::lowest();
        double stagnationEpsilon = 1e-6;
        double oldCost = numeric_limits<double>::lowest();
        vector<double> cost;
        vector<double> cost1;

        double mu_path = mu_k;


        //start main loop
        while(!converged) {
            MatrixXd Y_k_A = X_k_A + ((t_km1 - 1) / t_k) * (X_k_A - X_km1_A);
            MatrixXd Y_k_E = X_k_E + ((t_km1 - 1) / t_k) * (X_k_E - X_km1_E);

            double rankA = 0.0, cardE = 0.0;

            if (option.lineSearhFlag) {
                MatrixXd G_k_A = Y_k_A - (1 / tau_k) * (Y_k_A + Y_k_E - D);
                MatrixXd G_k_E = Y_k_E - (1 / tau_k) * (Y_k_A + Y_k_E - D);

                JacobiSVD<MatrixXd> svd(G_k_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto diagS = svd.singularValues();

                MatrixXd posM1 = posMat(diagS.array() - mu_k / tau_k).asDiagonal();
                MatrixXd X_kp1_A = svd.matrixU() * posM1 * svd.matrixV().transpose();

                MatrixXd posM2 = posMat(G_k_E.array().abs() - option.lambda * mu_k / tau_k);
                MatrixXd X_kp1_E = signMat(G_k_E) * posM2;
            } else {

            }
        }

    }

}//namespace dynamic_stereo