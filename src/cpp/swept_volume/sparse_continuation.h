#ifndef SC
#define SC
#include <Eigen/Core>
#include <iostream>
#include <vector>
void sparse_continuation(const Eigen::RowVector3d p0, const std::vector<Eigen::RowVector3i> init_voxels, const std::vector<double> t0, const std::function<double(const Eigen::RowVector3d &, double &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &)> scalarFunc, const double eps, const int expected_number_of_cubes, Eigen::VectorXd & CS, Eigen::MatrixXd & CV, Eigen::MatrixXi & CI, Eigen::VectorXd & CV_argmins_vector);


void sparse_continuation(const Eigen::RowVector3d p0, const std::vector<Eigen::RowVector3i> init_voxels, const std::vector<Eigen::RowVectorXd> t0, const  std::function<double(const Eigen::RowVector3d &, Eigen::RowVectorXd &, std::vector<std::vector<Eigen::RowVectorXd>> &, std::vector<std::vector<double>> &, std::vector<std::vector<Eigen::RowVectorXd>> &)> scalarFunc, const double eps, const int expected_number_of_cubes, Eigen::VectorXd & CS, Eigen::MatrixXd & CV, Eigen::MatrixXi & CI, Eigen::MatrixXd & CV_argmins_vector);

#endif
