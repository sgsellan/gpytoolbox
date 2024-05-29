#ifndef OUTSIDE_POINTS_FROM_RASTERIZATION_H
#define OUTSIDE_POINTS_FROM_RASTERIZATION_H

#include <Eigen/Core>

template<int dim>
void outside_points_from_rasterization(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    const int res,
    const double tol,
    const bool narrow_band,
    const bool parallel,
    const bool force_cpu,
    const bool verbose,
    Eigen::MatrixXd & outside_points);

#endif