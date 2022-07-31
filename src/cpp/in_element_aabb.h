#ifndef IN_ELEMENT_AABB
#define IN_ELEMENT_AABB

#include <Eigen/Core>

void in_element_aabb(const Eigen::MatrixXd& queries, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXi & I);

#endif