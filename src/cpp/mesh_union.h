#ifndef MESHUNION
#define MESHUNION

#include <Eigen/Core>

void mesh_union(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& V2, const Eigen::MatrixXi& F2, Eigen::MatrixXd& V3, Eigen::MatrixXi& F1);

#endif