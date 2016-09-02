//
// Created by yanhang on 9/2/16.
//

#include "RPCA.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{

    static double matNorm(const MatrixXd& m){

    }

    void solveRPCA(const Eigen::MatrixXd& D, Eigen::MatrixXd& A_hat, Eigen::MatrixXd& E_hat, int& numIter,
                   const RPCAOption& option){
        const int maxLineSearchIter = 200;
        const int m = D.rows();
        const int n = D.cols();

        //initialization
        double t_k = 1, t_km1 = 1, tau_9 = 2;
        MatrixXd X_km1_A = MatrixXd::Zero(m, n);
        MatrixXd X_km1_E = MatrixXd::Zero(m, n);
        MatrixXd X_k_A = MatrixXd::Zero(m, n);
        MatrixXd X_k_E  = MatrixXd::Zero(m, n);

        double mu_k = option.mu, mu_0, mu_bar;
        if(option.continuationFlag){
            if(option.lineSearhFlag){
                mu_0 = option.eta *
            }
        }
    }

}//namespace dynamic_stereo