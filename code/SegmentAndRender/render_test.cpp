//
// Created by yanhang on 9/4/16.
//

#include "RPCA.h"
#include "gtest/gtest.h"

using namespace std;
using namespace Eigen;
using namespace dynamic_stereo;

TEST(RPCA, RPCA){
    MatrixXd D = MatrixXd::Ones(10,10);
    D(4,4) = 7;
    D(1,5) = 8;
    MatrixXd A_hat, E_hat;
    int numIter;

    RPCAOption option;
    option.lambda = sqrt(10);
    option.maxIter = 10;
    option.lineSearhFlag = true;

    solveRPCA(D, A_hat, E_hat, numIter, option);

    cout << "A_hat:" << endl << A_hat << endl;
    cout << "numIter:" << numIter << endl;
}


