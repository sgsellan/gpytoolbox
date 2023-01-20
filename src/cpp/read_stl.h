#ifndef READ_STL
#define READ_STL

#include <Eigen/Core>

int read_stl(
    const std::string& file,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);

#endif