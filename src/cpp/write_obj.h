#ifndef WRITE_OBJ
#define WRITE_OBJ

#include <Eigen/Core>

int write_obj(const std::string& file,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& Ft,
    const Eigen::MatrixXd& N,
    const Eigen::MatrixXi& Fn);

#endif

