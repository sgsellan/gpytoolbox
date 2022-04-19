#ifndef RAY_MESH_INTERSECT_AABB
#define RAY_MESH_INTERSECT_AABB

#include <Eigen/Core>

void ray_mesh_intersect_aabb(Eigen::MatrixXd & sources, Eigen::MatrixXd & directions,Eigen::MatrixXd & VT, Eigen::MatrixXi & FT, Eigen::VectorXd & ts, Eigen::VectorXi & ids, Eigen::MatrixXd & lambdas);

#endif