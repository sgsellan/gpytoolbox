#ifndef RNDONMESH
#define RNDONMESH
#include <Eigen/Core>

void random_points_on_mesh(const int n, const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, Eigen::MatrixXd & X, Eigen::MatrixXd & N);

#endif
