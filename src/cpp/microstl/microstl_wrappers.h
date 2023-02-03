#ifndef MICROSTL_WRAPPERS_H
#define MICROSTL_WRAPPERS_H

#include <Eigen/Core>

int read_stl(
    const std::string& file,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);

int write_stl(
    const std::string& file,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const bool binary);

#endif

