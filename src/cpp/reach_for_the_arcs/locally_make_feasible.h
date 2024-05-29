#ifndef LOCALLY_MAKE_FEASIBLE_H
#define LOCALLY_MAKE_FEASIBLE_H

#include <Eigen/Core>

template<int dim>
void locally_make_feasible(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    const Eigen::MatrixXd & outside_points,
    const Eigen::VectorXi & batch,
    const int rng_seed,
    const int n_local_searches,
    const int local_search_iters,
    const double tol,
    const double clamp_value,
    const bool parallel,
    const bool verbose,
    Eigen::MatrixXd & cloud_points,
    Eigen::MatrixXd & cloud_normals,
    Eigen::MatrixXi & feasible);

#endif