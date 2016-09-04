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
        MatrixXd res(m);
#pragma omp parallel for
        for(auto y=0; y<m.rows(); ++y){
            for(auto x=0; x<m.cols(); ++x){
                if(m(y,x) < 0)
                    res(y,x) = 0;
            }
        }
        return res;
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
        return res;
    }

    void solveRPCA(const Eigen::MatrixXd& D, Eigen::MatrixXd& A_hat, Eigen::MatrixXd& E_hat, int& numIter,
                   const RPCAOption& option){
        const int maxLineSearchIter = 200;
        const int m = D.rows();
        const int n = D.cols();

        double lambda = option.lambda;
        if(lambda < 0)
            lambda = std::sqrt(std::max(m, n));

        //initialization
        double t_k = 1, t_km1 = 1, tau_0 = 2;
        MatrixXd X_km1_A = MatrixXd::Zero(m, n);
        MatrixXd X_km1_E = MatrixXd::Zero(m, n);
        MatrixXd X_k_A = MatrixXd::Zero(m, n);
        MatrixXd X_k_E  = MatrixXd::Zero(m, n);

        double mu_k = option.mu, mu_0, mu_bar;
        if(option.continuationFlag){
            double dn = math_util::matrix2Norm<MatrixXd>(D);
            if(option.lineSearhFlag)
                mu_0 = option.eta * dn;
            else
                mu_0 = dn;

            mu_k = 0.99 * mu_0;
            mu_bar = 1e-9 * mu_0;
        }


        double tau_k = tau_0;
        bool converged = false;
        numIter = 0;

        double stoppingCriterionOld = numeric_limits<double>::lowest();
        double stagnationEpsilon = 1e-6;
        double oldCost = numeric_limits<double>::lowest();

        double mu_path = mu_k;


        //start main loop
        while(!converged) {
            MatrixXd Y_k_A = X_k_A + ((t_km1 - 1) / t_k) * (X_k_A - X_km1_A);
            MatrixXd Y_k_E = X_k_E + ((t_km1 - 1) / t_k) * (X_k_E - X_km1_E);

            double rankA = 0.0, cardE = 0.0;

            MatrixXd X_kp1_A, X_kp1_E;
            if (!option.lineSearhFlag) {
                MatrixXd G_k_A = Y_k_A - (1 / tau_k) * (Y_k_A + Y_k_E - D);
                MatrixXd G_k_E = Y_k_E - (1 / tau_k) * (Y_k_A + Y_k_E - D);


                JacobiSVD<MatrixXd> svd(G_k_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto diagS = svd.singularValues();

                MatrixXd tmp1 = (diagS.array() - mu_k / tau_k);
                MatrixXd posM1 = posMat(tmp1);

                X_kp1_A = svd.matrixU() * posM1.asDiagonal() * svd.matrixV().transpose();

                MatrixXd tmp2 = G_k_E.array().abs() - lambda * mu_k / tau_k;
                MatrixXd posM2 = posMat(tmp2);
                X_kp1_E = signMat(G_k_E).array() * posM2.array();
            } else {
                bool convergedLineSearch = false;
                int numLineSearchIter = 0;

                double tau_hat = option.eta * tau_k;

                while(!convergedLineSearch){

                }
            }

            double t_kp1 = 0.5 * (1 + std::sqrt(1+4*t_k*t_k));

            MatrixXd temp = X_kp1_A + X_kp1_E - Y_k_A - Y_k_E;
            MatrixXd S_kp1_A = tau_k * (Y_k_A - X_kp1_A) + temp;
            MatrixXd S_kp1_E = tau_k * (Y_k_E - X_kp1_E) + temp;

            MatrixXd S_kp1_AE(S_kp1_A.rows(), S_kp1_A.cols() + S_kp1_E.cols());
            S_kp1_AE << S_kp1_A, S_kp1_E;
            MatrixXd X_kp1_AE(X_kp1_A.rows(), X_kp1_A.cols() * X_kp1_E.cols());
            X_kp1_AE << X_kp1_A, X_kp1_E;

            double stoppingCriterion = S_kp1_AE.norm() / (tau_k * std::max(1.0, X_kp1_AE.norm()));
            if(stoppingCriterion <= option.tol){
                converged = true;
            }

            if(option.continuationFlag)
                mu_k = std::max(0.9*mu_k, mu_bar);

            t_km1 = t_k;
            t_k = t_kp1;
            X_km1_A = X_k_A;
            X_km1_E = X_k_E;
            X_k_A = X_kp1_A;
            X_k_E = X_kp1_E;
            numIter += 1;
            if(!converged && numIter >= option.maxIter)
                converged = true;
        }

        A_hat = X_k_A;
        E_hat = X_k_E;
    }

}//namespace dynamic_stereo