//
// Created by yanhang on 9/2/16.
// A C++ implementation of RPCA using proximal gradient
//

#ifndef DYNAMICSTEREO_RPCA_H
#define DYNAMICSTEREO_RPCA_H

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

namespace dynamic_stereo{

    struct RPCAOption{
        double lambda;
        int maxIter;
        double tol;
        bool lineSearhFlag;
        bool continuationFlag;
        double eta;
        double mu;
        bool verbose;
        RPCAOption(): lambda(-1), maxIter(10000), tol(1e-7), lineSearhFlag(false), continuationFlag(true),
                      eta(0.9), mu(1e-3), verbose(false){}
    };

    void solveRPCA(const Eigen::MatrixXd& D, Eigen::MatrixXd& A_hat, Eigen::MatrixXd& E_hat, int& numIter,
                   const RPCAOption& option = RPCAOption());

}//namespace dynamic_stereo


#endif //DYNAMICSTEREO_RPCA_H
