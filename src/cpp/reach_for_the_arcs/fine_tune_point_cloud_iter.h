#ifndef FINE_TUNE_POINT_CLOUD_ITER_H
#define FINE_TUNE_POINT_CLOUD_ITER_H

#include <Eigen/Core>

template<int dim>
void fine_tune_point_cloud_iter(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    const Eigen::MatrixXd & reconstructed_V,
    const Eigen::MatrixXi & reconstructed_F,
    Eigen::MatrixXd & cloud_points,
    Eigen::MatrixXd & cloud_normals,
    Eigen::MatrixXi & feasible,
    const Eigen::VectorXi & batch,
    const int max_points_per_sphere,
    const double rng_seed,
    const int n_local_searches,
    const int local_search_iters,
    const double local_search_t,
    const double tol,
    const double clamp_value,
    const bool parallel,
    const bool verbose);

#endif
