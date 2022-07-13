#ifndef TANGENTIAL_RELAXATION
#define TANGENTIAL_RELAXATION



#include <Eigen/Core>

void tangential_relaxation(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature,
Eigen::MatrixXd & V0 ,Eigen::MatrixXi & F0, Eigen::VectorXd & lambda);


#endif
