#ifndef READ_OBJ
#define READ_OBJ

#include <Eigen/Core>

int read_obj(const std::string& file,
    const bool return_UV,
    const bool return_N,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& UV,
    Eigen::MatrixXi& Ft,
    Eigen::MatrixXd& N,
    Eigen::MatrixXi& Fn);

#endif

