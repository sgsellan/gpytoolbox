#ifndef IN_ELEMENT_AABB
#define IN_ELEMENT_AABB

#include <Eigen/Core>
#include <Eigen/Sparse>

void in_element_aabb(Eigen::MatrixXd & queries, Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::VectorXi & I);

#endif