#ifndef SWEPT_VOLUME_FUN
#define SWEPT_VOLUME_FUN
#include <Eigen/Core>
#include <vector>

void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd transformation_matrix, const double eps, const int num_seeds, const bool verbose, Eigen::MatrixXd & U, Eigen::MatrixXi & G);

#endif
