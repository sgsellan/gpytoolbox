#ifndef TINYPLY_WRAPPERS_H
#define TINYPLY_WRAPPERS_H

#include <Eigen/Core>

int read_ply(
    const std::string& file,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);

int write_ply(
    const std::string& file,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const bool binary);

#endif

