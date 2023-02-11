#ifndef TINYPLY_WRAPPERS_H
#define TINYPLY_WRAPPERS_H

#include <Eigen/Core>

int read_ply(
    const std::string& filepath,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& N,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& C);

int write_ply(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& N,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& C,
    const bool binary);

#endif

